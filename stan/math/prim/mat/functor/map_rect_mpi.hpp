#pragma once

#include <vector>
#include <map>
#include <type_traits>

#include <stan/math/prim/mat/fun/to_array_1d.hpp>
#include <stan/math/prim/arr/functor/mpi_command.hpp>
#include <stan/math/prim/arr/functor/mpi_cluster.hpp>
#include <stan/math/prim/mat/fun/dims.hpp>

namespace stan {
  namespace math {

    template <typename ReduceF, typename CombineF>
    class mpi_parallel_call;

    namespace internal {

      template <int member, typename T>
      class mpi_parallel_call_cache {
        // static members to hold locally cached data
        // of placing the cache inside mpi_parallel_call is that we will
        // cache the data multiple times (once for each ReduceF
        // type).
        typedef const T cache_t;
        static std::map<int, cache_t> local_;

        template <typename, typename>
        friend class stan::math::mpi_parallel_call;
      };
      
      template <int member, typename T>
      std::map<int, const T>
      mpi_parallel_call_cache<member, T>::local_;
    }

    // utility class to store and cache static data and output size
    // per job; manages memory allocation and cluster communication
    template <typename ReduceF, typename CombineF>
    class mpi_parallel_call {
      boost::mpi::communicator world_;
      const std::size_t rank_ = world_.rank();
      const std::size_t world_size_ = world_.size();

      // local caches which hold local slices of data
      typedef internal::mpi_parallel_call_cache<1, std::vector<std::vector<double>>> t_cache_x_r;
      typedef internal::mpi_parallel_call_cache<2, std::vector<std::vector<int>>> t_cache_x_i;
      typedef internal::mpi_parallel_call_cache<3, std::vector<int>> t_cache_f_out;
      typedef internal::mpi_parallel_call_cache<4, std::vector<int>> t_cache_chunks;
      
      const int callsite_id_;

      const CombineF combine_;

      typedef typename CombineF::result_type result_type;

      vector_d local_shared_params_dbl_;
      matrix_d local_job_params_dbl_;

      template <typename T_cache>
      typename T_cache::cache_t& cache_lookup() { return(T_cache::local_.find(callsite_id_)->second); }
      
      template <typename T_cache>
      bool cache_contains() { return(T_cache::local_.count(callsite_id_) == 1); }
      
      template <typename T_cache>
      void cache_store(typename T_cache::cache_t element) { T_cache::local_.insert(std::make_pair(callsite_id_, element)); }
      
    public:
      // called on root
      template <typename T_shared_param, typename T_job_param>
      mpi_parallel_call(const Eigen::Matrix<T_shared_param, Eigen::Dynamic, 1>& shared_params,
                        const std::vector<Eigen::Matrix<T_job_param, Eigen::Dynamic, 1>>& job_params,
                        const std::vector<std::vector<double>>& x_r,
                        const std::vector<std::vector<int>>& x_i,
                        int callsite_id)
        : callsite_id_(callsite_id),
          combine_(shared_params, job_params) {
        if(rank_ != 0)
          throw std::runtime_error("problem sizes can only defined on the root.");

        // make childs aware of upcoming job
        mpi_broadcast_command<stan::math::mpi_distributed_apply<mpi_parallel_call<ReduceF,CombineF>>>();

        // send callsite id
        boost::mpi::broadcast(world_, callsite_id, 0);

        std::vector<int> job_dims = dims(job_params);

        const size_type num_jobs = job_dims[0];
        const size_type num_job_params = job_dims[1];
        
        vector_d shared_params_dbl = value_of(shared_params);
        matrix_d job_params_dbl(num_job_params, num_jobs);

        for(std::size_t j=0; j != num_jobs; ++j)
          job_params_dbl.col(j) = value_of(job_params[j]);

        setup_call(shared_params_dbl, job_params_dbl, x_r, x_i);
      }

      // called on remote sites
      mpi_parallel_call(int callsite_id) : callsite_id_(callsite_id), combine_() {
        if(rank_ == 0)
          throw std::runtime_error("problem sizes must be defined on the root.");

        setup_call(vector_d(), matrix_d(), std::vector<std::vector<double>>(), std::vector<std::vector<int>>());
      }

      static void distributed_apply() {
        // entry point when called on remote
        int callsite_id;
        boost::mpi::communicator world;
        boost::mpi::broadcast(world, callsite_id, 0);

        //std::cout << "Starting on remote " << world.rank() << "; callsite_id = " << callsite_id << std::endl;

        // call constructor for the remotes
        mpi_parallel_call<ReduceF,CombineF> job_chunk(callsite_id);

        job_chunk.reduce();
      }

      // all data is cached and local parameters are also available
      result_type reduce() {

        //std::cout << "starting reduce on remote " << world_.rank() << "; callsite_id = " << callsite_id_ << std::endl;
        
        const std::vector<int>& job_chunks = cache_lookup<t_cache_chunks>();
        const int num_jobs = sum(job_chunks);

        // id of first job out of all
        int start_job = 0;
        for(std::size_t n=0; n != rank_; ++n)
          start_job += job_chunks[n];

        const int num_local_jobs = local_job_params_dbl_.cols();
        const std::size_t num_outputs_per_job = ReduceF::get_output_size(local_shared_params_dbl_.rows(), local_job_params_dbl_.rows());
        matrix_d local_output(num_outputs_per_job, num_local_jobs);
        std::vector<int> local_f_out(num_local_jobs, -1);

        t_cache_x_r::cache_t& local_x_r = cache_lookup<t_cache_x_r>();
        t_cache_x_i::cache_t& local_x_i = cache_lookup<t_cache_x_i>();

        // check if we know already output sizes
        if(cache_contains<t_cache_f_out>()) {
          t_cache_f_out::cache_t& f_out = cache_lookup<t_cache_f_out>();
          int num_outputs = 0;
          for(std::size_t j=start_job; j != start_job + num_local_jobs; ++j)
            num_outputs += f_out[j];
          local_output.resize(num_outputs_per_job, num_outputs);
        }

        int offset = 0;

        //std::cout << "evaluating " << num_local_jobs << " on remote " << world_.rank() << "; callsite_id = " << callsite_id_ << std::endl;
        
        try {
          for(std::size_t i=0; i != num_local_jobs; ++i) {
            const matrix_d job_output = ReduceF::apply(local_shared_params_dbl_, local_job_params_dbl_.col(i), local_x_r[i], local_x_i[i]);
            local_f_out[i] = job_output.cols();
          
            if(local_output.cols() < offset + local_f_out[i])
              local_output.conservativeResize(Eigen::NoChange, 2*(offset + local_f_out[i]));

            local_output.block(0, offset, local_output.rows(), local_f_out[i]) = job_output;

            offset += local_f_out[i];
          }
        } catch(const std::exception& e) {
          // We abort processing only and flag that things went
          // wrong. We have to keep processing to keep the cluster in
          // sync and let the gather_outputs method detect on the root
          // that things went wrong
          //std::cout << "CAUGHT EXCEPTION on " << rank_ << std::endl;
          local_output(0,offset) = std::numeric_limits<double>::max();
          //local_f_out[num_local_jobs-1] = -1;
        }

        // during first execution we distribute the output sizes from
        // local jobs to the root
        if(!cache_contains<t_cache_f_out>()) {
          std::vector<int> world_f_out(num_jobs, 0);
          boost::mpi::gatherv(world_, local_f_out.data(), num_local_jobs, world_f_out.data(), job_chunks, 0);
          // on the root we now have all sizes from all childs. Copy
          // over the local sizes to the world vector on each local
          // node in order to cache this information locally.
          std::copy(local_f_out.begin(), local_f_out.end(), world_f_out.begin() + start_job);
          // Before we can cache these sizes locally we must ensure
          // that no exception has been fired from any node. Hence,
          // check on the root that everything was ok and broadcast
          // that info. Only then we locally cache the output sizes.
          bool all_ok = true;
          for(std::size_t i=0; i < num_jobs; ++i)
            if(world_f_out[i] == -1)
              all_ok = false;
          boost::mpi::broadcast(world_, all_ok, 0);
          //std::cout << "On rank = " << rank_ << "; all_ok = " << all_ok << std::endl;
          if(!all_ok) {
            // err out on the root
            if (rank_ == 0)
              throw std::runtime_error("MPI error on first evaluation.");
            // and ensure on the workers that they return into their
            // listening state
            return(result_type());
          }
          cache_store<t_cache_f_out>(world_f_out);
        }

        t_cache_f_out::cache_t& world_f_out = cache_lookup<t_cache_f_out>();

        // check that cached sizes are the same as just collected from
        // this evaluation

        //std::cout << "gathering outputs " << sum(local_f_out) << " on remote " << world_.rank() << "; callsite_id = " << callsite_id_ << std::endl;

        return combine_.gather_outputs(local_output, world_f_out, job_chunks);
      }

    private:

      template <typename T_cache>
      void scatter_array_2d_cached(typename T_cache::cache_t& data) {
        // distribute data only if not in cache yet
        if(cache_contains<T_cache>()) {
          //std::cout << "cache_data on remote " << rank_ << " HIT the cache " << std::endl;
          return;
        }

        // number of elements of each data item must be transferred to
        // the workers
        std::vector<int> data_dims = dims(data);
        data_dims.resize(2);

        boost::mpi::broadcast(world_, data_dims.data(), 2, 0);

        const std::vector<int> job_chunks = mpi_map_chunks(data_dims[0], 1);
        const std::vector<int> data_chunks = mpi_map_chunks(data_dims[0], data_dims[1]);

        auto flat_data = to_array_1d(data);

        //std::cout << "cache_const_data on remote " << rank_ << " input size " << flat_data.size()<< std::endl;
        //std::cout << "cache_const_data on remote " << rank_ << " expecting size " << data_chunks[rank_] << std::endl;

        // transfer it ...
        decltype(flat_data) local_flat_data(data_chunks[rank_]);
        boost::mpi::scatterv(world_, flat_data.data(), data_chunks, local_flat_data.data(), 0);

        //std::cout << "cache_const_data on remote " << rank_ << " got scatterd data of size " << local_flat_data.size() << std::endl;

         // ... and rebuild 2D array
        std::vector<decltype(flat_data)> local_data;
        auto local_iter = local_flat_data.begin();
        for(std::size_t i=0; i != job_chunks[rank_]; ++i) {
          typename T_cache::cache_t::value_type const data_elem(local_iter, local_iter + data_dims[1]);
          local_data.push_back(data_elem);
          local_iter += data_dims[1];
        }

        // finally we cache it locally
        cache_store<T_cache>(local_data);
      }

      template <typename T_cache>
      void broadcast_1d_cached(typename T_cache::cache_t& data) {
        if(cache_contains<T_cache>()) {
          return;
        }

        std::size_t data_size = data.size();
        boost::mpi::broadcast(world_, data_size, 0);

        typename std::remove_cv<typename T_cache::cache_t>::type local_data = data;
        local_data.resize(data_size);
        
        boost::mpi::broadcast(world_, local_data.data(), data_size, 0);
        cache_store<T_cache>(local_data);
      }

      template <int meta_cache_id>
      vector_d broadcast_vector(const vector_d& data) {
        typedef internal::mpi_parallel_call_cache<meta_cache_id, std::vector<size_type>> t_meta_cache;
        std::vector<size_type> meta_info = { data.size() };
        broadcast_1d_cached<t_meta_cache>(meta_info);

        const std::vector<size_type>& data_size = cache_lookup<t_meta_cache>();

        vector_d local_data = data;
        local_data.resize(data_size[0]);
        
        boost::mpi::broadcast(world_, local_data.data(), data_size[0], 0);
        
        return(local_data);
      }


      // scatters an Eigen matrix column wise over the cluster. Meta
      // information as the data size is treated as static data and
      // only transferred on the first call and read from cache
      // subsequently.
      template <int meta_cache_id>
      matrix_d scatter_matrix(const matrix_d& data) {
        typedef internal::mpi_parallel_call_cache<meta_cache_id, std::vector<size_type>> t_meta_cache;
        std::vector<size_type> meta_info = { data.rows(), data.cols() };
        broadcast_1d_cached<t_meta_cache>(meta_info);

        const std::vector<size_type>& dims = cache_lookup<t_meta_cache>();
        const size_type rows = dims[0];
        const size_type total_cols = dims[1];

        const std::vector<int> job_chunks = mpi_map_chunks(total_cols, 1);
        const std::vector<int> data_chunks = mpi_map_chunks(total_cols, rows);
        matrix_d local_data(rows, job_chunks[rank_]);
        boost::mpi::scatterv(world_, data.data(), data_chunks, local_data.data(), 0);

        return(local_data);
      }
       
      void setup_call(const vector_d& shared_params, const matrix_d& job_params,
                      const std::vector<std::vector<double>>& x_r,
                      const std::vector<std::vector<int>>& x_i) {

        //std::cout << "setup_call on remote " << rank_ << "; callsite_id_ = " << callsite_id_ << std::endl;

        std::vector<int> job_chunks = mpi_map_chunks(job_params.cols(), 1);
        broadcast_1d_cached<t_cache_chunks>(job_chunks);
 
        local_shared_params_dbl_ = broadcast_vector<-1>(shared_params);
        local_job_params_dbl_ = scatter_matrix<-2>(job_params);

        //std::cout << "rank_ = " << rank_ << " got shared params:" << std::endl << local_shared_params_dbl_ << std::endl;
        //std::cout << "rank_ = " << rank_ << " got job params:" << std::endl << local_job_params_dbl_ << std::endl;
        
        // distribute const data if not yet cached
        scatter_array_2d_cached<t_cache_x_r>(x_r);
        //std::cout << "setup_call on remote " << rank_ << "; got real data" << std::endl;
        scatter_array_2d_cached<t_cache_x_i>(x_i);
        //std::cout << "setup_call on remote " << rank_ << "; got int data" << std::endl;
      }

    };

    template <typename F, typename T_shared_param, typename T_job_param>
    struct map_rect_reduce;

    template <typename F>
    struct map_rect_reduce<F, double, double> {
      static std::size_t get_output_size(std::size_t num_shared_params, std::size_t num_job_specific_params) {
        return(1);
      }
      static matrix_d apply(const vector_d& shared_params, const vector_d& job_specific_params, const std::vector<double>& x_r, const std::vector<int>& x_i) {
        const vector_d out = F()(shared_params, job_specific_params, x_r, x_i);
        return( out.transpose() );
      }
    };
      
    template <typename F, typename T_shared_param, typename T_job_param>
    class map_rect_combine {
      boost::mpi::communicator world_;
      const std::size_t rank_ = world_.rank();
      const std::size_t world_size_ = world_.size();

      const Eigen::Matrix<T_shared_param, Eigen::Dynamic, 1>* shared_params_operands_;
      const std::vector<Eigen::Matrix<T_job_param, Eigen::Dynamic, 1>>* job_params_operands_;
      
    public:

      typedef Eigen::Matrix<typename stan::return_type<T_shared_param, T_job_param>::type, Eigen::Dynamic, 1> result_type;
      
      map_rect_combine() {}
      map_rect_combine(const Eigen::Matrix<T_shared_param, Eigen::Dynamic, 1>& shared_params,
                       const std::vector<Eigen::Matrix<T_job_param, Eigen::Dynamic, 1>>& job_params)
        : shared_params_operands_(&shared_params), job_params_operands_(&job_params) {}

      result_type gather_outputs(const matrix_d& local_result, const std::vector<int>& world_f_out, const std::vector<int>& job_chunks) const {

        const std::size_t num_jobs = world_f_out.size();
        const std::size_t num_output_size_per_job = local_result.rows();
        const std::size_t size_world_f_out = sum(world_f_out);

        matrix_d world_result(num_output_size_per_job, size_world_f_out);

        std::vector<int> chunks_result(world_size_,0);
        for(std::size_t i=0, ij=0; i != world_size_; ++i)
          for(std::size_t j=0; j != job_chunks[i]; ++j, ++ij)
            chunks_result[i] += world_f_out[ij] * num_output_size_per_job;

        // collect results
        boost::mpi::gatherv(world_, local_result.data(), chunks_result[rank_], world_result.data(), chunks_result, 0);

        // only add result to AD tree on root
        if(rank_ != 0)
          return(result_type());

        result_type out(size_world_f_out);

        const std::size_t num_shared_operands = shared_params_operands_->size();
        const std::size_t num_job_operands = (*job_params_operands_)[0].size();

        const std::size_t offset_job_params = is_constant_struct<T_shared_param>::value ? 1 : 1+num_shared_operands ;

        for(std::size_t i=0, ij=0; i != num_jobs; ++i) {
          for(std::size_t j=0; j != world_f_out[i]; ++j, ++ij) {
            // check if the outputs flags a failure
            if(unlikely(world_result(0,ij) == std::numeric_limits<double>::max()))
              throw std::runtime_error("MPI error.");

            operands_and_partials<Eigen::Matrix<T_shared_param, Eigen::Dynamic, 1>,
                                  Eigen::Matrix<T_job_param, Eigen::Dynamic, 1> >
              ops_partials(*shared_params_operands_, (*job_params_operands_)[i]);

            if (!is_constant_struct<T_shared_param>::value)
              ops_partials.edge1_.partials_ = world_result.block(1,ij,num_shared_operands,1);
              
            if (!is_constant_struct<T_job_param>::value)
                ops_partials.edge2_.partials_ = world_result.block(offset_job_params,ij,num_job_operands,1);

            out(ij) = ops_partials.build(world_result(0,ij));
          }
        }

        return(out);
      }
    };

    template <typename F, typename T_shared_param, typename T_job_param>
    Eigen::Matrix<typename stan::return_type<T_shared_param, T_job_param>::type, Eigen::Dynamic, 1>
    map_rect_mpi(const Eigen::Matrix<T_shared_param, Eigen::Dynamic, 1>& shared_params,
                 const std::vector<Eigen::Matrix<T_job_param, Eigen::Dynamic, 1>>& job_params,
                 const std::vector<std::vector<double>>& x_r,
                 const std::vector<std::vector<int>>& x_i,
                 const int callsite_id) {
      typedef map_rect_reduce<F, T_shared_param, T_job_param> ReduceF;
      typedef map_rect_combine<F, T_shared_param, T_job_param> CombineF;
      
      mpi_parallel_call<ReduceF,CombineF> job_chunk(shared_params, job_params, x_r, x_i, callsite_id);

      return job_chunk.reduce();
    }
  }
}