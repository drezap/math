#pragma once

#include <vector>
#include <set>

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <stan/math/prim/arr/functor/map_rect_mpi.hpp>

namespace stan {
  namespace math {

    namespace internal {
            
      template <typename F>
      class distributed_map_rect {
        boost::mpi::communicator world_;
        const std::size_t R_ = world_.rank();
        const std::size_t W_ = world_.size();

        // local version of things in flattened out form
        std::vector<double> local_theta_;
        std::size_t uid_;

        // note: it would be nice to have these const, but I think C++
        // doesn't let me given how it has to be initialized
        std::size_t N_; // # of jobs for world
        std::size_t T_; // # of parameters
        std::size_t X_r_; // # of real data items per job
        std::size_t X_i_; // # of int  data items per job
        
        // # of outputs per job
        std::size_t F_out_sum_ = 0;
        double *final_result_ = 0;
        std::vector<int> world_F_out_;

        // the parameters which can be updated; only stored at the
        // root
        typedef const std::vector<std::vector<var> > param_t;
        param_t* theta_;

        std::vector<int> chunks_;

        // # of jobs for this chunk
        std::size_t C_;

      public:
        // called on root, sents problem sizes, data and parameters
        distributed_map_rect(const std::vector<std::vector<var> >& theta,
                             const std::vector<std::vector<double> >& x_r,
                             const std::vector<std::vector<int> >& x_i,
                             const std::size_t uid)
          : uid_(uid), N_(theta.size()), T_(theta[0].size()), X_r_(x_r[0].size()), X_i_(x_i[0].size()) {
          //std::cout << "Setting up distributed_map on root " << world_.rank() << " / " << W_ << std::endl;
          if(R_ != 0)
            throw std::runtime_error("problem sizes can only defined on the root.");

          // checks if the data is already cached
          distribute_map_rect_data(x_r, x_i, uid);

          //std::cout << "root uses UID = " << uid_ << std::endl;

          // make childs aware of upcoming job
          mpi_cluster::broadcast_command<stan::math::distributed_apply<distributed_map_rect<F> > >();

          //std::cout << "setting up root with uid = " << uid << std::endl;
          setup(uid);
          
          // sent uid and # of params
          std::vector<int> meta(2, 0);
          meta[0] = uid;
          meta[1] = T_ ;
          boost::mpi::broadcast(world_, meta.data(), 2, 0);
          
          distribute_param(theta);

          (*this)();
        }

        distributed_map_rect() {
          //std::cout << "Setting up distributed map on worker " << world_.rank() << " / " << W_ << std::endl;
          if(R_ == 0)
            throw std::runtime_error("problem sizes must be defined on the root.");

          //std::cout << "setting up child ..." << std::endl;
          
          // get uid & # of params from root
          std::vector<int> meta(2);
          boost::mpi::broadcast(world_, meta.data(), 2, 0);
          uid_ = meta[0];
          T_ = meta[1];

          //std::cout << "worker " << world_.rank() << " / " << W_ << " uses UID = " << uid_ << std::endl;

          setup(uid_);
          
          distribute_param(std::vector<std::vector<var> >());

          (*this)();
        }
        
        static void distributed_apply() {
          // entry point when called on remote

          // call default constructor
          distributed_map_rect<F> job_chunk;
        }

        // process work chunks; collect results at root
        void operator()() {
          using std::vector;

          const distributed_map_rect_data& local_ = distributed_data::get_const_instance().find(uid_)->second;
          
          vector<double> local_result;
          // reserve output size for 5 function outputs per job... we
          // will find out during evaluation
          local_result.reserve(C_*5*(1+T_));

          // number of outputs per job
          vector<int> local_F_out(C_, 0);

          try {
            for(std::size_t i = 0; i != C_; i++) {
              start_nested();

              // note: on the root node we could avoid re-creating
              // these theta var instances
              const vector<var> theta_run_v(local_theta_.begin() + i * T_, local_theta_.begin() + (i+1) * T_);

              const vector<var> fx_v = F::apply(theta_run_v, local_.x_r_[i], local_.x_i_[i]);
              const std::size_t F_out = fx_v.size();
              
              local_F_out[i] = F_out;

              vector<double> FJx;
              FJx.reserve(F_out * (T_+1));
              
              for (std::size_t i = 0; i != F_out; ++i) {
                FJx.push_back(fx_v[i].val());
                set_zero_all_adjoints_nested();
                grad(fx_v[i].vi_);
                for (std::size_t k = 0; k != T_; ++k)
                  FJx.push_back(theta_run_v[k].adj());
              }
              local_result.insert(local_result.end(), FJx.begin(), FJx.end());
              
              recover_memory_nested();
            }
          } catch(const std::exception& e) {
            recover_memory_nested();
            // we only abort further processing. The root node will
            // detect unfinished runs and throw. Flag failure.
            local_F_out = vector<int>(C_, 0);
            local_F_out[0] = -1;
            // we can discard all data
            local_result.resize(0);
          }
          // collect results at root
          //std::cout << "gathering output sizes..." << std::endl;
          boost::mpi::gatherv(world_, local_F_out.data(), C_, world_F_out_.data(), chunks_, 0);

          // find out cumulative size of output
          bool all_ok = true;
          F_out_sum_ = 0;

          // we only allocate any memory on the root
          if(R_ == 0) {
            for(std::size_t i=0; i != N_; ++i) {
              if(world_F_out_[i] == -1) {
                all_ok = false;
                world_F_out_[i] = 0; // set to 0 to get a correct sum below
              } else {
                F_out_sum_ += world_F_out_[i];
              }
            }
      
            // allocate memory on AD stack for final result. Note that the
            // gradients and the function results will land there
            final_result_
              = ChainableStack::memalloc_.alloc_array<double>( (T_ + 1) * F_out_sum_ );

          }

          vector<int> chunks_result(W_,0);
          for(std::size_t i=0, ij=0; i != W_; ++i)
            for(std::size_t j=0; j != chunks_[i]; ++j, ++ij)
              chunks_result[i] += world_F_out_[ij] * (T_ + 1);

          // collect results
          //std::cout << "gathering actual outputs..." << std::endl;
          boost::mpi::gatherv(world_, local_result.data(), local_result.size(), final_result_, chunks_result, 0);

          // now we can throw on the root if necessary as all workers have finished
          if(unlikely(R_ == 0 & !all_ok))
            throw std::runtime_error("MPI error");

          //std::cout << "results have been send!" << std::endl;
        }

        // called only on the root to register results in the AD stack
        // and setup a the results var vector
        std::vector<var> register_results() {
          if(R_ != 0)
              throw std::runtime_error("results must be registered on root only.");

          //std::cout << "registering results" << std::endl;
          
          // insert results into the AD tree
          vari** varis
            = ChainableStack::memalloc_.alloc_array<vari*>(N_*T_);

          std::vector<var> res;
          res.reserve(F_out_sum_);
    
          for(std::size_t i=0, ik=0; i != N_; i++) {
            // link the operands...
            for(std::size_t j=0; j != T_; j++)
              varis[i * T_ + j] = (*theta_)[i][j].vi_;

            // ...with partials of outputs
            for(std::size_t k=0; k != world_F_out_[i]; k++, ik++) {
              const double val = *(final_result_ + (T_+1) * ik);
              res.push_back(var(new precomputed_gradients_vari(val, T_, varis + i * T_, final_result_ + (T_+1) * ik + 1)));
            }
          }
      
          //std::cout << "results are registered" << std::endl;

          return(res);
        }

        std::size_t get_uid() const {
          return uid_;
        }

      private:
        // sent a new set of parameters out
        void distribute_param(const std::vector<std::vector<var> >& theta) {
          if(R_ == 0) {
            if(theta.size() != N_)
              throw std::runtime_error("problem size mismatch.");
            if(theta[0].size() != T_)
              throw std::runtime_error("parameter size mismatch.");

            theta_ = &theta;
          }

          // flatten and send out/recieve using scatterv
          std::vector<double> world_theta;
          world_theta.reserve(theta.size()*T_);
          
          for(std::size_t n = 0; n != theta.size(); ++n) {
            const std::vector<double> theta_n_d = stan::math::value_of(theta[n]);
            world_theta.insert(world_theta.end(), theta_n_d.begin(), theta_n_d.end());
          }
          
          const std::vector<int> chunks_theta = mpi_cluster::map_chunks(N_, T_);
          local_theta_.resize(chunks_theta[R_]);
          boost::mpi::scatterv(world_, world_theta.data(), chunks_theta, local_theta_.data(), 0);
          //std::cout << "Job " << R_ << " got " << local_theta_.size() << " parameters " << std::endl;
        }

        void setup(std::size_t uid) {
           // grab data
          //local_ = distributed_data::get_mutable_instance().find(uid)->second;
          const distributed_map_rect_data& local_ = distributed_data::get_const_instance().find(uid)->second;

          // copy over sizes, etc.
          N_ = local_.N_;
          X_r_ = local_.x_r_[0].size();
          X_i_ = local_.x_i_[0].size();

          chunks_ = mpi_cluster::map_chunks(N_, 1);
          world_F_out_ = std::vector<int>(N_, 0);
          C_ = chunks_[R_];

          //std::cout << "worker " << world_.rank() << " / " << W_ << " got shapes " << N_ << ", " << T_ << ", " << X_i_ << ", " << X_r_ << std::endl;
          
       }
      };      
      
    }

    /* an example user functor in Stan could be */
    // real[] map_rect(F f, real[,] theta, real[,] x_r, int[,] x_i);

    template <typename F>
    std::vector<var>
    map_rect_mpi(const std::vector<std::vector<var> >& theta,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 const std::size_t uid) {

      internal::distributed_map_rect<F> root_job_chunk(theta, x_r, x_i, uid);

      return(root_job_chunk.register_results());
    }
      
  }
}