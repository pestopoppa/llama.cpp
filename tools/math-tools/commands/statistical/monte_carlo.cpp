// Monte Carlo simulation and integration
// Supports: integration, expectation, variance, pi estimation, custom distributions

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include "../../include/xoshiro.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

namespace math_tools {

class MonteCarloCommand : public Command {
public:
    std::string name() const override { return "monte_carlo"; }

    std::string description() const override {
        return "Monte Carlo: integration, expectation, variance estimation";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            std::string operation = params.value("operation", "integrate");
            int n_samples = params.value("n_samples", 10000);
            uint64_t seed = params.value("seed", 42);

            if (operation == "integrate") {
                result = integrate(params, n_samples, seed);
            } else if (operation == "pi") {
                result = estimate_pi(n_samples, seed);
            } else if (operation == "normal_stats") {
                result = normal_stats(params, n_samples, seed);
            } else if (operation == "random_walk") {
                result = random_walk(params, seed);
            } else if (operation == "bootstrap") {
                result = bootstrap(params, n_samples, seed);
            } else {
                result.error = "Unknown operation: " + operation;
            }

        } catch (const std::exception& e) {
            result.error = e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    // Monte Carlo integration over [a, b]^d
    Result integrate(const json& params, int n_samples, uint64_t seed) {
        Result result;

        if (!params.contains("bounds")) {
            result.error = "integrate requires 'bounds' [[a1,b1], [a2,b2], ...]";
            return result;
        }

        auto bounds = params["bounds"].get<std::vector<std::vector<double>>>();
        int dim = bounds.size();

        // Calculate volume
        double volume = 1.0;
        for (const auto& b : bounds) {
            volume *= (b[1] - b[0]);
        }

        std::string func_type = params.value("function", "sphere");

        // Function to integrate
        std::function<double(const std::vector<double>&)> f;

        if (func_type == "sphere") {
            // Indicator function for unit sphere: 1 if ||x|| <= 1
            f = [](const std::vector<double>& x) -> double {
                double r2 = 0.0;
                for (double xi : x) {
                    r2 += xi * xi;
                }
                return r2 <= 1.0 ? 1.0 : 0.0;
            };
        } else if (func_type == "gaussian") {
            // Multivariate standard Gaussian
            f = [](const std::vector<double>& x) -> double {
                double r2 = 0.0;
                for (double xi : x) {
                    r2 += xi * xi;
                }
                return std::exp(-0.5 * r2);
            };
        } else if (func_type == "polynomial") {
            // Sum of squares: f(x) = sum(x_i^2)
            f = [](const std::vector<double>& x) -> double {
                double sum = 0.0;
                for (double xi : x) {
                    sum += xi * xi;
                }
                return sum;
            };
        } else {
            // Default: constant 1 (volume estimation)
            f = [](const std::vector<double>&) -> double { return 1.0; };
        }

        // Sample and integrate
        double sum = 0.0;
        double sum_sq = 0.0;

#ifdef HAVE_OPENMP
        #pragma omp parallel reduction(+:sum,sum_sq)
        {
            Xoshiro256pp rng(seed + omp_get_thread_num());
            #pragma omp for
            for (int i = 0; i < n_samples; ++i) {
                std::vector<double> x(dim);
                for (int d = 0; d < dim; ++d) {
                    x[d] = rng.uniform(bounds[d][0], bounds[d][1]);
                }
                double fx = f(x);
                sum += fx;
                sum_sq += fx * fx;
            }
        }
#else
        Xoshiro256pp rng(seed);
        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> x(dim);
            for (int d = 0; d < dim; ++d) {
                x[d] = rng.uniform(bounds[d][0], bounds[d][1]);
            }
            double fx = f(x);
            sum += fx;
            sum_sq += fx * fx;
        }
#endif

        double mean = sum / n_samples;
        double variance = (sum_sq / n_samples) - mean * mean;
        double std_error = std::sqrt(variance / n_samples);

        double integral = volume * mean;
        double error = volume * std_error;

        json data;
        data["integral"] = integral;
        data["standard_error"] = error;
        data["n_samples"] = n_samples;
        data["volume"] = volume;
        data["mean_value"] = mean;
        data["variance"] = variance;

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Classic pi estimation using circle area
    Result estimate_pi(int n_samples, uint64_t seed) {
        Result result;

        int inside = 0;

#ifdef HAVE_OPENMP
        #pragma omp parallel reduction(+:inside)
        {
            Xoshiro256pp rng(seed + omp_get_thread_num());
            #pragma omp for
            for (int i = 0; i < n_samples; ++i) {
                double x = rng.uniform(-1.0, 1.0);
                double y = rng.uniform(-1.0, 1.0);
                if (x*x + y*y <= 1.0) {
                    inside++;
                }
            }
        }
#else
        Xoshiro256pp rng(seed);
        for (int i = 0; i < n_samples; ++i) {
            double x = rng.uniform(-1.0, 1.0);
            double y = rng.uniform(-1.0, 1.0);
            if (x*x + y*y <= 1.0) {
                inside++;
            }
        }
#endif

        double pi_estimate = 4.0 * inside / n_samples;
        double error = std::abs(pi_estimate - M_PI);
        double std_error = 4.0 * std::sqrt((double)inside * (n_samples - inside) / n_samples) / n_samples;

        json data;
        data["pi_estimate"] = pi_estimate;
        data["true_pi"] = M_PI;
        data["absolute_error"] = error;
        data["standard_error"] = std_error;
        data["n_samples"] = n_samples;
        data["inside_count"] = inside;

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Statistics of a normal distribution
    Result normal_stats(const json& params, int n_samples, uint64_t seed) {
        Result result;

        double mu = params.value("mean", 0.0);
        double sigma = params.value("std", 1.0);

        std::vector<double> samples(n_samples);
        Xoshiro256pp rng(seed);

        double sum = 0.0;
        double sum_sq = 0.0;
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();

        for (int i = 0; i < n_samples; ++i) {
            double x = rng.normal(mu, sigma);
            samples[i] = x;
            sum += x;
            sum_sq += x * x;
            min_val = std::min(min_val, x);
            max_val = std::max(max_val, x);
        }

        double sample_mean = sum / n_samples;
        double sample_var = (sum_sq / n_samples) - sample_mean * sample_mean;

        // Sort for percentiles
        std::sort(samples.begin(), samples.end());

        json data;
        data["sample_mean"] = sample_mean;
        data["sample_std"] = std::sqrt(sample_var);
        data["min"] = min_val;
        data["max"] = max_val;
        data["median"] = samples[n_samples / 2];
        data["percentile_25"] = samples[n_samples / 4];
        data["percentile_75"] = samples[3 * n_samples / 4];
        data["n_samples"] = n_samples;
        data["true_mean"] = mu;
        data["true_std"] = sigma;

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Random walk simulation
    Result random_walk(const json& params, uint64_t seed) {
        Result result;

        int n_steps = params.value("n_steps", 1000);
        int n_walks = params.value("n_walks", 100);
        double step_size = params.value("step_size", 1.0);
        std::string walk_type = params.value("type", "1d");

        std::vector<double> final_positions(n_walks);
        std::vector<double> final_distances(n_walks);

        Xoshiro256pp rng(seed);

        for (int w = 0; w < n_walks; ++w) {
            if (walk_type == "1d") {
                double pos = 0.0;
                for (int s = 0; s < n_steps; ++s) {
                    pos += (rng.uniform() < 0.5 ? -step_size : step_size);
                }
                final_positions[w] = pos;
                final_distances[w] = std::abs(pos);
            } else {
                // 2D walk
                double x = 0.0, y = 0.0;
                for (int s = 0; s < n_steps; ++s) {
                    double angle = rng.uniform(0, 2 * M_PI);
                    x += step_size * std::cos(angle);
                    y += step_size * std::sin(angle);
                }
                final_distances[w] = std::sqrt(x*x + y*y);
            }
        }

        // Statistics
        double mean_dist = std::accumulate(final_distances.begin(), final_distances.end(), 0.0) / n_walks;
        double mean_pos = std::accumulate(final_positions.begin(), final_positions.end(), 0.0) / n_walks;

        // Expected: sqrt(n_steps) * step_size for 1D
        double expected_rms = std::sqrt(n_steps) * step_size;

        json data;
        data["mean_final_distance"] = mean_dist;
        data["expected_rms_distance"] = expected_rms;
        data["n_walks"] = n_walks;
        data["n_steps"] = n_steps;
        data["type"] = walk_type;

        if (walk_type == "1d") {
            data["mean_final_position"] = mean_pos;
        }

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Bootstrap resampling for confidence intervals
    Result bootstrap(const json& params, int n_samples, uint64_t seed) {
        Result result;

        if (!params.contains("data")) {
            result.error = "bootstrap requires 'data' array";
            return result;
        }

        auto data_vec = params["data"].get<std::vector<double>>();
        int n = data_vec.size();

        std::string statistic = params.value("statistic", "mean");
        double confidence = params.value("confidence", 0.95);

        // Bootstrap resampling
        std::vector<double> bootstrap_stats(n_samples);
        Xoshiro256pp rng(seed);

        for (int b = 0; b < n_samples; ++b) {
            // Resample with replacement
            std::vector<double> resample(n);
            for (int i = 0; i < n; ++i) {
                int idx = static_cast<int>(rng.uniform() * n);
                resample[i] = data_vec[idx];
            }

            // Compute statistic
            if (statistic == "mean") {
                bootstrap_stats[b] = std::accumulate(resample.begin(), resample.end(), 0.0) / n;
            } else if (statistic == "median") {
                std::sort(resample.begin(), resample.end());
                bootstrap_stats[b] = resample[n / 2];
            } else if (statistic == "std") {
                double mean = std::accumulate(resample.begin(), resample.end(), 0.0) / n;
                double var = 0.0;
                for (double x : resample) {
                    var += (x - mean) * (x - mean);
                }
                bootstrap_stats[b] = std::sqrt(var / n);
            }
        }

        // Sort for percentiles
        std::sort(bootstrap_stats.begin(), bootstrap_stats.end());

        double alpha = (1.0 - confidence) / 2;
        int lo_idx = static_cast<int>(alpha * n_samples);
        int hi_idx = static_cast<int>((1 - alpha) * n_samples);

        // Original statistic
        double original_stat;
        if (statistic == "mean") {
            original_stat = std::accumulate(data_vec.begin(), data_vec.end(), 0.0) / n;
        } else if (statistic == "median") {
            std::vector<double> sorted_data = data_vec;
            std::sort(sorted_data.begin(), sorted_data.end());
            original_stat = sorted_data[n / 2];
        } else {
            double mean = std::accumulate(data_vec.begin(), data_vec.end(), 0.0) / n;
            double var = 0.0;
            for (double x : data_vec) {
                var += (x - mean) * (x - mean);
            }
            original_stat = std::sqrt(var / n);
        }

        json out;
        out["statistic"] = statistic;
        out["original_value"] = original_stat;
        out["confidence_interval"]["lower"] = bootstrap_stats[lo_idx];
        out["confidence_interval"]["upper"] = bootstrap_stats[hi_idx];
        out["confidence_level"] = confidence;
        out["n_bootstrap_samples"] = n_samples;
        out["n_data_points"] = n;

        result.success = true;
        result.data = out.dump();
        return result;
    }
};

std::unique_ptr<Command> create_monte_carlo() {
    return std::make_unique<MonteCarloCommand>();
}

} // namespace math_tools
