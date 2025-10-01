#include <iostream>
#include <cmath>
#include <stdexcept>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include "arima_modeling.h"

using namespace std;


arma::vec acf(const std::vector<SunspotEntry>& data, int maxlag) {
    int n = data.size();
    arma::vec x(n);
    for (int i = 0; i < n; i++) {
        x(i) = data[i].SNvalue;
    }

    arma::vec acf(maxlag+1);
    for (int k = 0; k <= maxlag; k++) {
        double sum = 0;
        for (int i = k; i < n; i++) {
            sum += x(i) * x(i-k);
        }
        acf(k) = sum / (n - k);
    }
    return acf;
}

arma::vec pacf(const std::vector<SunspotEntry>& data, int maxlag) {
    int n = data.size();
    arma::vec x(n);
    for (int i = 0; i < n; i++) {
        x(i) = data[i].SNvalue;
    }

    arma::vec pacf_vals(maxlag+1, arma::fill::zeros);
    arma::mat R(maxlag+1, maxlag+1, arma::fill::zeros);

    // Compute autocorrelations for different lags
    for (int i = 0; i <= maxlag; i++) {
        for (int j = i; j < n; j++) {
            R(i, 0) += x[j] * x[j - i];
        }
        R(i, 0) /= (n - i);
    }

    // Normalize the first value to 1
    pacf_vals(0) = 1.0;
    R(0, 0) = 1.0;

    // Apply the Durbin-Levinson algorithm to solve Yule-Walker equations
    for (int k = 1; k <= maxlag; k++) {
        double num = R(k, 0);
        for (int j = 1; j < k; j++) {
            num -= R(k - j, 0) * R(j, k - 1);
        }
        double den = 1.0;
        for (int j = 1; j < k; j++) {
            den -= R(j, 0) * R(j, k - 1);
        }
        pacf_vals(k) = num / den;

        for (int j = 1; j < k; j++) {
            R(j, k) = R(j, k - 1) - pacf_vals(k) * R(k - j, k - 1);
        }
        R(k, k) = pacf_vals(k);
    }

    return pacf_vals;
}

std::pair<int, int> get_pq(const std::vector<SunspotEntry>& data) {
    int max_order = 5;
    int n = data.size();
    cout << "n = " << n << endl;
    
    // Use a subset of data for faster computation
    int subset_size = std::min(n, 10000);
    std::vector<SunspotEntry> subset(data.end() - subset_size, data.end());
    
    arma::vec acf_vals = acf(subset, max_order);
    arma::vec pacf_vals = pacf(subset, max_order);
    double threshold = 1.96 / sqrt(subset_size);

    // Find optimal p (AR order) from PACF
    int best_p = 1; // Default to at least AR(1)
    for (int p = 1; p <= max_order; p++) {
        if (std::abs(pacf_vals(p)) < threshold) {
            best_p = p - 1;
            break;
        }
    }

    // Find optimal q (MA order) from ACF
    int best_q = 1; // Default to at least MA(1)
    for (int q = 1; q <= max_order; q++) {
        if (std::abs(acf_vals(q)) < threshold) {
            best_q = q - 1;
            break;
        }
    }

    // Ensure we don't select (0,0)
    if (best_p == 0 && best_q == 0) {
        best_p = 1;
    }

    cout << "Selected ARMA order: p=" << best_p << ", q=" << best_q << endl;
    return std::make_pair(best_p, best_q);
}
// Fit an ARMA model to a time series using maximum likelihood estimation
ARIMAModel fit_arma_model(const vector<SunspotEntry>& data, int p, int q) {
    int n = data.size();
    arma::vec data_vec(n);
    for (int i = 0; i < n; i++) {
        data_vec(i) = data[i].SNvalue;
    }

    // Initialize model struct
    ARIMAModel model;
    model.p = p;
    model.d = 0; // Initialize d to 0 (no differencing for now)
    model.q = q;
    model.phi = arma::vec(p, arma::fill::zeros);
    model.theta = arma::vec(q, arma::fill::zeros);

    // Handle the case where p=0 and q=0 (white noise model)
    if (p == 0 && q == 0) {
        model.residuals_var = arma::var(data_vec);
        model.log_likelihood = -n * log(sqrt(2 * M_PI * model.residuals_var)) - 0.5 * arma::sum(arma::pow(data_vec - arma::mean(data_vec), 2)) / model.residuals_var;
        model.aic = -2 * model.log_likelihood + 2;
        model.bic = -2 * model.log_likelihood + log(n);
        return model;
    }

    // For non-trivial models, use a simplified fitting approach
    int start_idx = std::max(p, q);
    arma::vec y = data_vec.subvec(start_idx, n-1);
    int m = y.n_elem;
    
    if (p > 0 && q == 0) {
        // Pure AR model
        arma::mat X(m, p);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                X(i, j) = data_vec(start_idx + i - j - 1);
            }
        }
        model.phi = arma::solve(X, y);
    } else if (p == 0 && q > 0) {
        // Pure MA model - simplified approach using sample ACF
        arma::vec acf_vals = acf(data, q);
        for (int i = 0; i < q; i++) {
            model.theta(i) = acf_vals(i+1) * 0.5; // Simplified estimation
        }
    } else {
        // ARMA model - simplified approach
        arma::mat X(m, p + q);
        arma::vec residuals = data_vec - arma::mean(data_vec);
        
        for (int i = 0; i < m; i++) {
            // AR terms
            for (int j = 0; j < p; j++) {
                X(i, j) = data_vec(start_idx + i - j - 1);
            }
            // MA terms (using residuals approximation)
            for (int j = 0; j < q; j++) {
                if (start_idx + i - j - 1 >= 0) {
                    X(i, p + j) = residuals(start_idx + i - j - 1);
                } else {
                    X(i, p + j) = 0;
                }
            }
        }
        
        arma::vec coefficients = arma::solve(X, y);
        model.phi = coefficients.subvec(0, p-1);
        model.theta = coefficients.subvec(p, p+q-1);
    }

    // Compute residuals for model evaluation
    arma::vec residuals(n);
    for (int i = start_idx; i < n; i++) {
        double y_pred = 0;
        
        // AR component
        for (int j = 0; j < p; j++) {
            y_pred += model.phi(j) * data_vec(i - j - 1);
        }
        
        // MA component (simplified)
        for (int j = 0; j < q; j++) {
            if (i - j - 1 >= 0) {
                y_pred += model.theta(j) * (data_vec(i - j - 1) - arma::mean(data_vec));
            }
        }
        
        residuals(i) = data_vec(i) - y_pred;
    }

    // Compute model statistics
    model.residuals_var = arma::var(residuals.subvec(start_idx, n-1));
    double residual_sum_sq = arma::as_scalar(arma::sum(arma::pow(residuals.subvec(start_idx, n-1), 2)));
    model.log_likelihood = -m * log(sqrt(2 * M_PI * model.residuals_var)) - 0.5 * residual_sum_sq / model.residuals_var;
    
    int num_params = p + q + 1;
    model.aic = -2 * model.log_likelihood + 2 * num_params;
    model.bic = -2 * model.log_likelihood + num_params * log(m);

    // Print the model parameters and evaluation metrics
    cout << "ARMA(" << p << ", " << q << ") model:" << endl;
    cout << "AR coefficients: " << model.phi.t() << endl;
    cout << "MA coefficients: " << model.theta.t() << endl;
    cout << "Residuals variance: " << model.residuals_var << endl;
    cout << "Log likelihood: " << model.log_likelihood << endl;
    cout << "AIC: " << model.aic << endl;
    cout << "BIC: " << model.bic << endl;

    return model;
}

double evaluate(const std::vector<SunspotEntry>& data, ARIMAModel model) {
    int n = data.size();
    arma::vec phi(model.p), theta(model.q);
    double rmse;
    arma::vec predictions;
    predictions.resize(n);

    for (int i = model.p + model.q; i < n; i++) {
        double y_pred = 0;
        for (int j = 0; j < model.p; j++) {
            y_pred += phi(j) * data[i - j - 1].SNvalue;
        }
        for (int j = 0; j < model.q; j++) {
            y_pred += theta(j) * predictions[i - j - 1];
        }
        predictions(i) = y_pred;
    }

    arma::vec data_values(n);
    for (int i = 0; i < n; i++) {
        data_values(i) = data[i].SNvalue;
    }
    
    rmse = sqrt(arma::mean(arma::pow(predictions - data_values, 2)));

    return rmse;
}

// Forecast next n_periods using the fitted ARIMA model
std::vector<double> forecast(const std::vector<SunspotEntry>& data, const ARIMAModel& model, int n_periods) {
    int n = data.size();
    std::vector<double> forecasts(n_periods);
    std::vector<double> series_extended;
    
    // Copy original data
    for (int i = 0; i < n; i++) {
        series_extended.push_back(data[i].SNvalue);
    }
    
    // Calculate mean for centering
    double data_mean = 0;
    for (int i = 0; i < n; i++) {
        data_mean += data[i].SNvalue;
    }
    data_mean /= n;
    
    // Generate forecasts
    for (int h = 0; h < n_periods; h++) {
        double forecast_val = 0.0;
        
        // AR component
        for (int j = 0; j < model.p && j < (int)series_extended.size(); j++) {
            forecast_val += model.phi(j) * series_extended[series_extended.size() - 1 - j];
        }
        
        // MA component (using recent residuals approximation)
        for (int j = 0; j < model.q && (int)series_extended.size() > j; j++) {
            double residual_approx = series_extended[series_extended.size() - 1 - j] - data_mean;
            forecast_val += model.theta(j) * residual_approx;
        }
        
        // If no coefficients, use mean
        if (model.p == 0 && model.q == 0) {
            forecast_val = data_mean;
        }
        
        forecasts[h] = forecast_val;
        series_extended.push_back(forecast_val);
    }
    
    return forecasts;
}

// Split data into training and testing sets
std::pair<std::vector<SunspotEntry>, std::vector<SunspotEntry>> train_test_split(const std::vector<SunspotEntry>& data, double train_ratio) {
    int n = data.size();
    int train_size = static_cast<int>(n * train_ratio);
    
    std::vector<SunspotEntry> train_data(data.begin(), data.begin() + train_size);
    std::vector<SunspotEntry> test_data(data.begin() + train_size, data.end());
    
    return std::make_pair(train_data, test_data);
}

// Evaluate predictions against actual values
double evaluate_predictions(const std::vector<SunspotEntry>& actual, const std::vector<double>& predicted) {
    if (actual.size() != predicted.size()) {
        throw std::invalid_argument("Actual and predicted vectors must have the same size");
    }
    
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double error = actual[i].SNvalue - predicted[i];
        sum_squared_error += error * error;
    }
    
    return sqrt(sum_squared_error / actual.size());
}