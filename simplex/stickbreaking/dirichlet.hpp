
// Code generated by stanc 2803de5
#include <stan/model/model_header.hpp>
namespace dirichlet_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 19> locations_array__ = 
{" (found before start of program)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 6, column 2 to column 16)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 9, column 2 to column 15)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 10, column 2 to column 16)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 12, column 6 to column 40)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 13, column 6 to column 36)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 11, column 19 to line 14, column 7)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 11, column 2 to line 14, column 7)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 15, column 2 to column 25)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 19, column 6 to column 63)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 18, column 19 to line 20, column 5)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 18, column 2 to line 20, column 5)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 21, column 2 to column 39)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 2, column 2 to column 17)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 3, column 18 to column 19)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 3, column 2 to column 27)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 6, column 9 to column 12)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 9, column 10 to column 11)",
 " (in '/Users/meenaljhajharia/cmdstan/transforms/simplex/stickbreaking/dirichlet.stan', line 10, column 9 to column 12)"};




class dirichlet_model final : public model_base_crtp<dirichlet_model> {

 private:
  int K;
  Eigen::Matrix<double, -1, 1> alpha__;
  int Y_1dim__;
  int z_1dim__; 
  Eigen::Map<Eigen::Matrix<double, -1, 1>> alpha{nullptr, 0};
 
 public:
  ~dirichlet_model() { }
  
  inline std::string model_name() const final { return "dirichlet_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 2803de5", "stancflags = "};
  }
  
  
  dirichlet_model(stan::io::var_context& context__,
                  unsigned int random_seed__ = 0,
                  std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "dirichlet_model_namespace::dirichlet_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 13;
      context__.validate_dims("data initialization","K","int",
           std::vector<size_t>{});
      K = std::numeric_limits<int>::min();
      
      
      current_statement__ = 13;
      K = context__.vals_i("K")[(1 - 1)];
      current_statement__ = 13;
      stan::math::check_greater_or_equal(function__, "K", K, 0);
      current_statement__ = 14;
      stan::math::validate_non_negative_index("alpha", "K", K);
      current_statement__ = 15;
      context__.validate_dims("data initialization","alpha","double",
           std::vector<size_t>{static_cast<size_t>(K)});
      alpha__ = 
        Eigen::Matrix<double, -1, 1>::Constant(K,
          std::numeric_limits<double>::quiet_NaN());
      new (&alpha) Eigen::Map<Eigen::Matrix<double, -1, 1>>(alpha__.data(), K);
        
      
      {
        std::vector<local_scalar_t__> alpha_flat__;
        current_statement__ = 15;
        alpha_flat__ = context__.vals_r("alpha");
        current_statement__ = 15;
        pos__ = 1;
        current_statement__ = 15;
        for (int sym1__ = 1; sym1__ <= K; ++sym1__) {
          current_statement__ = 15;
          stan::model::assign(alpha, alpha_flat__[(pos__ - 1)],
            "assigning variable alpha", stan::model::index_uni(sym1__));
          current_statement__ = 15;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 15;
      stan::math::check_greater_or_equal(function__, "alpha", alpha, 0);
      current_statement__ = 16;
      Y_1dim__ = std::numeric_limits<int>::min();
      
      
      current_statement__ = 16;
      Y_1dim__ = (K - 1);
      current_statement__ = 16;
      stan::math::validate_non_negative_index("Y", "K - 1", Y_1dim__);
      current_statement__ = 17;
      stan::math::validate_non_negative_index("x", "K", K);
      current_statement__ = 18;
      z_1dim__ = std::numeric_limits<int>::min();
      
      
      current_statement__ = 18;
      z_1dim__ = (K - 1);
      current_statement__ = 18;
      stan::math::validate_non_negative_index("z", "K - 1", z_1dim__);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = Y_1dim__;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "dirichlet_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      Eigen::Matrix<local_scalar_t__, -1, 1> Y =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(Y_1dim__,
           DUMMY_VAR__);
      current_statement__ = 1;
      Y = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(
            Y_1dim__);
      Eigen::Matrix<local_scalar_t__, -1, 1> x =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(K, DUMMY_VAR__);
      Eigen::Matrix<local_scalar_t__, -1, 1> z =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(z_1dim__,
           DUMMY_VAR__);
      current_statement__ = 7;
      for (int k = 1; k <= (K - 1); ++k) {
        current_statement__ = 4;
        stan::model::assign(z,
          stan::math::inv_logit(
            (stan::model::rvalue(Y, "Y", stan::model::index_uni(k)) -
              stan::math::log((K - k)))),
          "assigning variable z", stan::model::index_uni(k));
        current_statement__ = 5;
        stan::model::assign(x,
          ((1 -
             stan::math::sum(
               stan::model::rvalue(x, "x",
                 stan::model::index_min_max(1, (k - 1))))) *
            stan::model::rvalue(z, "z", stan::model::index_uni(k))),
          "assigning variable x", stan::model::index_uni(k));
      }
      current_statement__ = 8;
      stan::model::assign(x,
        (1 -
          stan::math::sum(
            stan::model::rvalue(x, "x",
              stan::model::index_min_max(1, (K - 1))))),
        "assigning variable x", stan::model::index_uni(K));
      current_statement__ = 2;
      stan::math::check_simplex(function__, "x", x);
      {
        current_statement__ = 11;
        for (int k = 1; k <= (K - 1); ++k) {
          current_statement__ = 9;
          lp_accum__.add(
            ((stan::math::log(
                stan::model::rvalue(z, "z", stan::model::index_uni(k))) +
               stan::math::log1m(
                 stan::model::rvalue(z, "z", stan::model::index_uni(k)))) +
              stan::math::log1m(
                stan::math::sum(
                  stan::model::rvalue(x, "x",
                    stan::model::index_min_max(1, (k - 1)))))));
        }
        current_statement__ = 12;
        lp_accum__.add(stan::math::dirichlet_lpdf<propto__>(x, alpha));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "dirichlet_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      Eigen::Matrix<double, -1, 1> Y =
         Eigen::Matrix<double, -1, 1>::Constant(Y_1dim__,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      Y = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(
            Y_1dim__);
      Eigen::Matrix<double, -1, 1> x =
         Eigen::Matrix<double, -1, 1>::Constant(K,
           std::numeric_limits<double>::quiet_NaN());
      Eigen::Matrix<double, -1, 1> z =
         Eigen::Matrix<double, -1, 1>::Constant(z_1dim__,
           std::numeric_limits<double>::quiet_NaN());
      out__.write(Y);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 7;
      for (int k = 1; k <= (K - 1); ++k) {
        current_statement__ = 4;
        stan::model::assign(z,
          stan::math::inv_logit(
            (stan::model::rvalue(Y, "Y", stan::model::index_uni(k)) -
              stan::math::log((K - k)))),
          "assigning variable z", stan::model::index_uni(k));
        current_statement__ = 5;
        stan::model::assign(x,
          ((1 -
             stan::math::sum(
               stan::model::rvalue(x, "x",
                 stan::model::index_min_max(1, (k - 1))))) *
            stan::model::rvalue(z, "z", stan::model::index_uni(k))),
          "assigning variable x", stan::model::index_uni(k));
      }
      current_statement__ = 8;
      stan::model::assign(x,
        (1 -
          stan::math::sum(
            stan::model::rvalue(x, "x",
              stan::model::index_min_max(1, (K - 1))))),
        "assigning variable x", stan::model::index_uni(K));
      current_statement__ = 2;
      stan::math::check_simplex(function__, "x", x);
      if (emit_transformed_parameters__) {
        out__.write(x);
        out__.write(z);
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      Eigen::Matrix<local_scalar_t__, -1, 1> Y =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(Y_1dim__,
           DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= Y_1dim__; ++sym1__) {
        stan::model::assign(Y, in__.read<local_scalar_t__>(),
          "assigning variable Y", stan::model::index_uni(sym1__));
      }
      out__.write(Y);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"Y", "x", "z"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{
                                                                   static_cast<size_t>(Y_1dim__)
                                                                   },
      std::vector<size_t>{static_cast<size_t>(K)},
      std::vector<size_t>{static_cast<size_t>(z_1dim__)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= Y_1dim__; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "Y" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= K; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "x" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= z_1dim__; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "z" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= Y_1dim__; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "Y" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= (K - 1); ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "x" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= z_1dim__; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "z" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"Y\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(Y_1dim__) + "},\"block\":\"parameters\"},{\"name\":\"x\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(K) + "},\"block\":\"transformed_parameters\"},{\"name\":\"z\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(z_1dim__) + "},\"block\":\"transformed_parameters\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"Y\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(Y_1dim__) + "},\"block\":\"parameters\"},{\"name\":\"x\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string((K - 1)) + "},\"block\":\"transformed_parameters\"},{\"name\":\"z\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(z_1dim__) + "},\"block\":\"transformed_parameters\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = Y_1dim__;
      const size_t num_transformed = emit_transformed_parameters * 
  (K + z_1dim__);
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      std::vector<int> params_i;
      vars = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = Y_1dim__;
      const size_t num_transformed = emit_transformed_parameters * 
  (K + z_1dim__);
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      vars = std::vector<double>(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 1> names__{"Y"};
      const std::array<Eigen::Index, 1> constrain_param_sizes__{Y_1dim__};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}
using stan_model = dirichlet_model_namespace::dirichlet_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return dirichlet_model_namespace::profiles__;
}

#endif


