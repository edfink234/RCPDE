#include <iostream>
#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <filesystem> // For checking file existence (C++17 and later)
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <random>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// Constants for the potential
constexpr double m = 1.0;           // Mass
constexpr double Omega = 0.2;       // Frequency of the harmonic trap
constexpr double A = 1.0;           // Amplitude of the Gaussian potential
constexpr double sigma = 1.0;       // Width of the Gaussian potential
constexpr double T = 10.0;          // Final time
constexpr double dt = 0.01;         // Time step
constexpr double x_star = 0.0;      // Final position sought
constexpr double v_th = 0.01;       // Velocity threshold, not used currently
constexpr double x_th = 0.01;       // Position threshold, not used currently

constexpr bool load_model = true;
constexpr bool automate = false;
constexpr bool produceInverse = true;
constexpr bool saveLibTorch = true;

// Path to the weights file
std::string weight_file = "../../NeuralNetworkData/xi_model_IC_2_point_225840410642715_.pt";

std::unordered_map<std::string, int> to_time = {{"timed", 0}, {"time", 3600}};
auto start_time = std::time(nullptr);

// Penalty factors
constexpr double smoothness_penalty_factor = 1e-3;
constexpr double time_penalty_factor = 1e-3;
constexpr double velocity_penalty = 1e-3;
constexpr double xi_penalty = 1e-3;
double x_start_val = 0.0;
constexpr double v_start_val = 0.0; // Initial velocity

auto x_start = torch::tensor({{x_start_val}}, torch::dtype(torch::kFloat32));
auto v_start = torch::tensor({{v_start_val}}, torch::dtype(torch::kFloat32));

constexpr double learning_rate = 1; //learning rate
torch::jit::script::Module myModule; //the model for xi(t)
// Generate t_values and delta_t
auto t_values = torch::linspace(1e-8, T, static_cast<int>(T / dt));
auto delta_t = t_values[1].item<double>() - t_values[0].item<double>();

// Helper function to replace '.' with "_point_" in a string representation of a double
std::string flt_to_str(double flt)
{
    std::string str = std::to_string(flt);
    std::replace(str.begin(), str.end(), '.', '_');
    return str;
}

// Extend torch with the sech function
namespace torch
{
    inline Tensor sech(const Tensor& x)
    {
        return 1.0 / torch::cosh(x);
    }
}

bool criterion()
{
    return !to_time["timed"] || (std::time(nullptr) - start_time < to_time["time"]);
}

// Compute the force (negative derivative of potential)
torch::Tensor force(const torch::Tensor& x, const torch::Tensor& xi)
{
    auto temp_arg = A * (x - xi);
    auto temp = torch::sech(temp_arg);
    return -(Omega * Omega * x) +
           (2 * A * A * A * temp * temp *
            torch::tanh(temp_arg));
}

// Derivative of position (dx/dt)
torch::Tensor dxdt(const torch::Tensor& v)
{
    return v;
}

// Derivative of velocity (dv/dt)
torch::Tensor dvdt(const torch::Tensor& x, const torch::Tensor& xi)
{
    return force(x, xi) / m;
}

// RK4 step for updating state
std::pair<torch::Tensor, torch::Tensor> rk4_step(const torch::Tensor& x, const torch::Tensor& v, const torch::Tensor& xi_t, double dt)
{
    // Ensure all inputs are tensors that require gradients
    auto x_grad = x.requires_grad() ? x : x.requires_grad_(true);
    auto v_grad = v.requires_grad() ? v : v.requires_grad_(true);
    auto xi_grad = xi_t.requires_grad() ? xi_t : xi_t.requires_grad_(true);

    // Calculate k1
    auto k1_x = dxdt(v_grad);
    auto k1_v = dvdt(x_grad, xi_grad);

    // Calculate k2
    auto k2_x = dxdt(v_grad + 0.5 * dt * k1_v);
    auto k2_v = dvdt(x_grad + 0.5 * dt * k1_x, xi_grad);

    // Calculate k3
    auto k3_x = dxdt(v_grad + 0.5 * dt * k2_v);
    auto k3_v = dvdt(x_grad + 0.5 * dt * k2_x, xi_grad);

    // Calculate k4
    auto k4_x = dxdt(v_grad + dt * k3_v);
    auto k4_v = dvdt(x_grad + dt * k3_x, xi_grad);

    // Update x and v using RK4 formula
    auto x_new = x_grad + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
    auto v_new = v_grad + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);

    // Return updated state
    return std::make_pair(x_new, v_new);
}


// Function to compute xi(t) using the model
torch::Tensor xi(double t)
{
    try
    {
        // Prepare the input tensor
        auto t_input = torch::tensor({{t}}, torch::dtype(torch::kFloat32));
        
        // Perform inference using the module's forward method
        auto output = myModule.forward({t_input}).toTensor();
        
        // Return the first scalar value of the output tensor
        return output;
    }
    catch (const c10::Error& e)
    {
        // Handle errors during inference
        std::cerr << "Error during inference: " << e.what() << '\n';
        throw std::runtime_error("Inference failed");
    }
}

// Loss function
std::pair<torch::Tensor, double> loss_func()
{
    // Initialization
    auto x = x_start;
    auto v = v_start;
    double smoothness_penalty = 0.0;
    std::vector<double> xi_values_temp;
    std::vector<double> v_values_temp;
    double best_loss = std::numeric_limits<double>::infinity();
    double best_time = std::numeric_limits<double>::infinity();
    size_t best_time_idx = std::numeric_limits<size_t>::max();

    // Loop over time values
    assert(t_values.size(0) == 1000);
    for (size_t i = 0; i < t_values.size(0); ++i)
    {
        double t = t_values[i].item<double>();
        auto xi_t = xi(t); // Compute xi(t)
        xi_values_temp.push_back(xi_t.item<double>()); // Store xi values
        auto step_result = rk4_step(x, v, xi_t, dt);
        x = step_result.first;
        v = step_result.second;
//        std::cout << "i = " << i << ", x = " << x.item<double>() << ", v = " << v.item<double>() << "\nt = " << t << ", xi(t) = " << xi_values_temp.back() << "\n\n";
        v_values_temp.push_back(std::abs(v.item<double>()));

        // Compute loss terms
        auto x_star_x_diff = (x_star - x);
        auto x_star_xi_diff = (x_star - xi(t));
        auto xi_0 = xi(0);
//        double x_star_x_diff = (x_star - x.item<double>());
//        double x_star_xi_diff = (x_star - xi(t).item<double>());
//        double xi_0 = xi(0).item<double>();
//        double v_val = v.item<double>();
        if (i > 1)
        {
            auto MSE = x_star_x_diff*x_star_x_diff + v*v + x_star_xi_diff*x_star_xi_diff + xi_0*xi_0;
//            double MSE = x_star_x_diff*x_star_x_diff + v_val*v_val + x_star_xi_diff*x_star_xi_diff + xi_0*xi_0;
            if (MSE.item<double>() < best_loss)
            {
//                std::cout << "x_star_x_diff*x_star_x_diff = " << x_star_x_diff*x_star_x_diff << '\n';
//                std::cout << "v*v = " << v*v << '\n';
//                std::cout << "x_star_xi_diff*x_star_xi_diff = " << x_star_xi_diff*x_star_xi_diff << '\n';
//                std::cout << "xi_0*xi_0 = " << xi_0*xi_0 << '\n';

                best_loss = MSE.item<double>();
                best_time = t;
                best_time_idx = i;
            }
        }
    }

    double v_best = v_values_temp[0];
    double xi_best = std::abs(xi_values_temp[0]);
    for (size_t i = 1; i <= best_time_idx; ++i)
    {
        auto delta_xi = xi_values_temp[i] - xi_values_temp[i - 1];
        auto derivative = delta_xi / delta_t;
        smoothness_penalty += derivative*derivative; // Penalty based on square of "derivative"
        if (v_values_temp[i] > v_best)
        {
            v_best = v_values_temp[i];
        }
        double abs_xi_temp_i = std::abs(xi_values_temp[i]);
        if (abs_xi_temp_i > xi_best)
        {
            xi_best = abs_xi_temp_i;
        }
    }

    smoothness_penalty /= static_cast<double>(best_time_idx);
//    std::cout << "best_loss = " << best_loss << ", smoothness_penalty = " << smoothness_penalty << "\nbest_time = " << best_time << ", v_best = " << v_best << "\nxi_best = " << xi_best << '\n';
    // Compute final loss
    auto loss = best_loss +
                smoothness_penalty_factor * smoothness_penalty +
                time_penalty_factor * best_time +
                velocity_penalty * v_best +
                xi_penalty * xi_best;

    return std::make_pair(torch::tensor(loss).requires_grad_(true), best_time);
}

// Convert module parameters to a vector of Tensors
std::vector<torch::Tensor> get_parameters(const torch::jit::script::Module& myModule)
{
    std::vector<torch::Tensor> params;
    for (const auto& p : myModule.parameters())
    {
        params.push_back(p);
    }
    return params;
}

int main()
{
    // Set seeds
    torch::manual_seed(42);
    
//    torch::Tensor tensor = torch::rand({2, 3});
//    std::cout << "Random Tensor:\n" << tensor << '\n';

    // Generate t_test_values
    auto t_test_values = torch::linspace(1e-8, T, static_cast<int>(T / dt));

    // Load initial condition from file
    std::ifstream file("../temp.txt");
    if (file.is_open())
    {
        file >> x_start_val;
        file.close();
    }
    else
    {
        std::cerr << "Unable to open temp.txt for reading initial condition." << '\n';
        return -1;
    }

    x_start = torch::tensor({{x_start_val}}, torch::dtype(torch::kFloat32));
    v_start = torch::tensor({{v_start_val}}, torch::dtype(torch::kFloat32));
    
    // Log initial conditions
    std::cout << "Initial conditions loaded:\n";
    std::cout << "x_start = " << x_start << '\n';
    std::cout << "v_start = " << v_start << '\n';

    // Check if the file exists
    if (!std::filesystem::exists(weight_file))
    {
        std::cerr << "Error: File " << weight_file << " not found!" << '\n';
        return -1; // Exit with error
    }
        
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        myModule = torch::jit::load(weight_file);
        puts("success!");
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model: " << e.what() << '\n';
        return -1;
    }
    
    // Prepare example input (matching the tracing input size/type)
    auto example = torch::tensor({{0.5}}, torch::dtype(torch::kFloat32));

    // Perform inference
    try
    {
        // Pass input to the module's forward method
        auto output = myModule.forward({example}).toTensor();
        
        // Print the output
        std::cout << "Model output: " << output << '\n';
        std::cout << "Model output using xi function: " << xi(example[0][0].item<double>()) << '\n';

    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error during inference: " << e.what() << '\n';
        return -1;
    }
    
    double best_loss = std::numeric_limits<double>::infinity();
    double best_t_value = 10.0; // T (final time)
    
    std::unordered_map<double, std::pair<double, double>> df;

    // Attempt to read the file
    file.open("../../dataFiles/ICs.txt");
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream ss(line);
            std::string value1, value2, value3;
            if (std::getline(ss, value1, ',') && std::getline(ss, value2, ',') && std::getline(ss, value3))
            {
                double key = std::stod(value1);
                double val1 = std::stod(value2);
                double val2 = std::stod(value3);
                df[key] = std::make_pair(val1, val2);
            }
        }
        file.close();
        std::cout << "Data read successfully:\n";
        for (const auto& [key, val] : df)
        {
            std::cout << key << ": (" << val.first << ", " << val.second << ")\n";
        }
    }
    else
    {
        // Create the file if it doesn't exist
        std::ofstream new_file("../dataFiles/ICs.txt");
        if (new_file.is_open())
        {
            new_file.close();
            std::cout << "File created successfully.\n";
        }
        else
        {
            std::cerr << "Failed to create the file.\n";
        }
    }

    double closest_x = x_start_val; // Assuming x_start is defined elsewhere
    if (df.find(x_start_val) != df.end())
    {
        best_loss = df[x_start_val].first;
        best_t_value = df[x_start_val].second;
        
        // Calculate the absolute differences
        auto differences = torch::abs(t_values - best_t_value);

        // Find the index of the minimum difference
        int64_t closest_index = differences.argmin().item<int64_t>();

        // Use the index to extract the desired slice
        t_test_values = t_values.index({torch::arange(0, closest_index + 1, torch::kInt64)});
//        std::cout << "t_test_values = " << t_test_values << '\n';
    }
    else
    {
        closest_x = std::numeric_limits<double>::infinity();
        for (const auto& [key, val] : df)
        {
            if (std::abs(key - x_start_val) < std::abs(closest_x - x_start_val))
            {
                closest_x = key;
            }
        }
    }

    std::cout << "Closest x: " << closest_x << "\n";
    std::cout << "Best loss: " << best_loss << "\n";
    std::cout << "Best t_value: " << best_t_value << "\n";
    
    auto test_loss = loss_func();
    std::cout << "Current loss = " << test_loss.first
    << "\ncurrent time = " << test_loss.second << '\n';

    if (!automate)
    {
        char ans;
        std::cout << "Proceed? (y/n): ";
        std::cin >> ans;
        if (ans != 'y')
        {
            exit(1);
        }
    }
    
    // Convert the parameters from the module
    std::vector<torch::Tensor> parameters = get_parameters(myModule);
    // Declare the Adam optimizer
    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(learning_rate)); 
    
//    std::cout << "t_test_values[-1].item<double>() = " << t_test_values[-1].item<double>() << '\n';
    
//    int num_params = 0, num_param = 0, random_param;
//    for (const auto& param: myModule.parameters())
//    {
//        num_params++;
//    }
//    std::cout << "num_params = " << num_params << '\n';
//    // Create a random device to obtain a seed for the random number engine
//    std::random_device rd;
//
//    // Create a Mersenne Twister engine with the seed
//    std::mt19937 gen(rd());
//
//    // Create a uniform integer distribution between 0 and num_params
//    std::uniform_int_distribution<> dis(0, num_params);
    
    long epoch = 0;
    while (criterion())
    {
        optimizer.zero_grad(); // Zero the gradients 
        // Compute the loss and time value
        auto [loss_value, t_value] = loss_func();
//        std::cout << myModule.parameters() << '\n';
        // Check if the current loss is the best (lowest)
        if (loss_value.item<double>() < best_loss)
        {
            best_loss = loss_value.item<double>();  // Update best loss
            best_t_value = t_value;  // Update corresponding best time

            // Save the model state
            myModule.save(weight_file);
            std::cout << "LibTorch version saved" << '\n';

            // Adjust `t_test_values` if `best_t_value` changed
            if (best_t_value != t_test_values[-1].item<double>())
            {
                std::cout << "New best t value = " << best_t_value << '\n';
                // Calculate the absolute differences
                auto differences = torch::abs(t_values - best_t_value);

                // Find the index of the minimum difference
                int64_t closest_index = differences.argmin().item<int64_t>();

                // Use the index to extract the desired slice
                t_test_values = t_values.index({torch::arange(0, closest_index + 1, torch::kInt64)});
            }

            std::cout << "Best model saved with loss: " << best_loss << '\n';
        }
        // Backpropagation
        loss_value.backward(); 
        torch::nn::utils::clip_grad_norm_(parameters, 1.0); // Clip gradients to avoid explosion
        optimizer.step(); // Update weights 

        std::cout << "Epoch " << epoch++ << ", Loss: " << loss_value.item<double>() << std::endl;

    }

    return 0;
}

/*
 cd build
 cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
 cmake --build . --config Release
 ./LibTorchExample
 */

