#include <iostream>
#include <vector>
#include <filesystem> // For checking file existence (C++17 and later)
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>


int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random Tensor:\n" << tensor << std::endl;
    // Create model instance
    // Path to the weights file
    std::string weight_file = "../traced_resnet_model.pt";

    // Check if the file exists
    if (!std::filesystem::exists(weight_file))
    {
        std::cerr << "Error: File " << weight_file << " not found!" << std::endl;
        return -1; // Exit with error
    }

    // Create model instance
//    XiModel model;
    
    torch::jit::script::Module myModule;
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
    auto input = torch::tensor({{0.5}}, torch::dtype(torch::kFloat32));

    // Perform inference
    try
    {
        // Pass input to the module's forward method
        auto output = myModule.forward({input}).toTensor();
        
        // Print the output
        std::cout << "Model output: " << output << '\n';
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error during inference: " << e.what() << '\n';
        return -1;
    }

    return 0;
}

/*
 cd build
 cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
 cmake --build . --config Release
 ./LibTorchExample
 */
