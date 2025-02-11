#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <future>         // std::async, std::future
#include <unordered_map>
#include <map>
#include <ctime>
#include <cstdlib>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cassert>
#include <thread>
#include <mutex>
#include <csignal>
#include <atomic>
#include <latch>
#include <tuple>
#include <functional>
#include <numbers>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <boost/unordered/concurrent_flat_map.hpp>

using Clock = std::chrono::high_resolution_clock;
std::atomic<bool> stop_flag(false);
const char* file_path = "../dataFiles/IC_expressions.txt";

void signalHandler(int signal)
{
    if (signal == SIGINT)
    {
        stop_flag.store(true);
        std::cout << "\nSIGINT received. Stopping...\n";
    }
}

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}

//https://medium.com/@ryan_forrester_/c-check-if-string-is-number-practical-guide-c7ba6db2febf
bool isFloat(const std::string& s)
{
    enum State { START, INT, FRAC, EXP, EXP_NUM };
    State state = START;
    bool has_digits = false;

    for (char c : s)
    {
        switch (state)
        {
            case START:
                if (c == '+' || c == '-') state = INT;
                else if (std::isdigit(c)) { state = INT; has_digits = true; }
                else if (c == '.') state = FRAC;
                else return false;
                break;
            case INT:
                if (std::isdigit(c)) has_digits = true;
                else if (c == '.') state = FRAC;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case FRAC:
                if (std::isdigit(c)) has_digits = true;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case EXP:
                if (c == '+' || c == '-' || std::isdigit(c)) state = EXP_NUM;
                else return false;
                break;
            case EXP_NUM:
                if (!std::isdigit(c)) return false;
                break;
        }
    }
    return has_digits && (state == INT || state == FRAC || state == EXP_NUM);
}

std::vector<std::string> split(const std::string& str)
{
    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string word;
    while (iss >> word)
    {
        result.push_back(word);
    }
    return result;
}

// Function to create a matrix with linspace columns. std::vector<float> min and
// std::vector<float> max must have size == cols
Eigen::MatrixXf createLinspaceMatrix(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));
    Eigen::MatrixXf mat(rows, cols);
    for (int col = 0; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            mat(row, col) = min_vec[col] + (max_vec[col] - min_vec[col]) * row / (rows - 1);
        }
    }
    return mat;
}

Eigen::MatrixXf generateData(int numRows, int numCols, float min = -3.0f, float max = 3.0f)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 thread_local gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    // Create the matrix
    Eigen::MatrixXf matrix(numRows, numCols);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            matrix(i, j) = distribution(gen);
        }
    }

    return matrix;
}

// Helper function to create a linspace vector
std::vector<float> linspace(float min_val, float max_val, int num_points)
{
    std::vector<float> linspaced(num_points);
    float step = (max_val - min_val) / (num_points - 1);
    for (int i = 0; i < num_points; ++i)
    {
        linspaced[i] = min_val + i * step;
    }
    return linspaced;
}

Eigen::MatrixXf createMeshgridWithLambda(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec, const std::function<float(const Eigen::RowVectorXf&)>& lambda)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));

    // Create linspaces for each variable (column)
    std::vector<std::vector<float>> linspaces;
    for (int col = 0; col < cols; ++col)
    {
        linspaces.push_back(linspace(min_vec[col], max_vec[col], rows)); // Assuming linspace function is defined elsewhere
    }

    // Calculate the total number of combinations (flattened meshgrid size)
    int total_combinations = 1;
    for (int col = 0; col < cols; ++col)
    {
        total_combinations *= rows;
    }

    // Create a matrix to store all combinations with the additional column for the lambda function result
    Eigen::MatrixXf matrix(total_combinations, cols + 1);

    // Fill in the matrix with all combinations of linspace values and apply the lambda function
    for (int col = 0; col < cols; ++col)
    {
        int repeat_count = 1;
        for (int i = col + 1; i < cols; ++i)
        {
            repeat_count *= rows;
        }

        int num_repeats = total_combinations / (repeat_count * rows);
        for (int repeat = 0; repeat < num_repeats; ++repeat)
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < repeat_count; ++j)
                {
                    int index = repeat * repeat_count * rows + i * repeat_count + j;
                    matrix(index, col) = linspaces[col][i];
                }
            }
        }
    }

    // Apply the lambda function to each row and store the result in the last column
    for (int i = 0; i < total_combinations; ++i)
    {
        matrix(i, cols) = lambda(matrix.row(i).head(cols));
    }

    return matrix;
}


Eigen::MatrixXf addColumnWithLambda(const Eigen::MatrixXf& matrix, const std::function<float(const Eigen::RowVectorXf&)>& lambda) {
    // Get the number of rows and columns of the input matrix
    int rows = matrix.rows();
    int cols = matrix.cols();
    
    // Create a new matrix with an additional column
    Eigen::MatrixXf newMatrix(rows, cols + 1);
    
    // Copy the original matrix into the new matrix (without the last column)
    newMatrix.block(0, 0, rows, cols) = matrix;
    
    // Apply the lambda function to each row and store the result in the last column
    for (int i = 0; i < rows; ++i) {
        newMatrix(i, cols) = lambda(matrix.row(i));
    }
    
    return newMatrix;
}

Eigen::MatrixXf createMeshgridVectors(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));

    // Create linspaces for each variable (column)
    std::vector<std::vector<float>> linspaces;
    for (int col = 0; col < cols; ++col)
    {
        linspaces.push_back(linspace(min_vec[col], max_vec[col], rows));
    }

    // Calculate the total number of combinations (flattened meshgrid size)
    int total_combinations = 1;
    for (int col = 0; col < cols; ++col)
    {
        total_combinations *= rows;
    }

    // Create a matrix to store all combinations (rows = total_combinations, cols = number of variables)
    Eigen::MatrixXf matrix(total_combinations, cols);

    // Fill in the matrix with all combinations of linspace values
    for (int col = 0; col < cols; ++col)
    {
        int repeat_count = 1;
        for (int i = col + 1; i < cols; ++i)
        {
            repeat_count *= rows;
        }
        
        int num_repeats = total_combinations / (repeat_count * rows);
        for (int repeat = 0; repeat < num_repeats; ++repeat)
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < repeat_count; ++j)
                {
                    int index = repeat * repeat_count * rows + i * repeat_count + j;
                    matrix(index, col) = linspaces[col][i];
                }
            }
        }
    }

    return matrix;
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

bool isInvalid(float x)
{
    return (std::isnan(x) || std::isinf(x));
}

bool isInvalid(const std::string& piece)
{
    for (int i = 0; i < static_cast<int>(piece.size()) - 2; i++)
    {
        if ((piece[i] == 'n' && piece[i+1] == 'a' && piece[i+2] == 'n') || (piece[i] == 'i' && piece[i+1] == 'n' && piece[i+2] == 'f'))
        {
            return true;
        }
    }
    return false;
}

bool isInvalid(const Eigen::VectorXf& vec)
{
    for (const auto& val: vec)
    {
        if (isInvalid(val))
        {
            return true;
        }
    }
    return false;
}

bool isInvalidExpression(const std::vector<std::string>& pieces)
{
    for (const std::string& piece: pieces)
    {
//        std::cout << "piece = " << piece << ", piece.size() = " << piece.size() << ", piece.size() - 2 = " << static_cast<int>(piece.size()) - 2 << '\n';
        if (isInvalid(piece))
        {
            return true;
        }
    }
    return false;
}

float Variance(const Eigen::VectorXf& vec)
{
    return (vec.array() - vec.mean()).square().sum() / vec.size();
}

class Data
{
    Eigen::MatrixXf data;
    std::unordered_map<std::string, Eigen::VectorXf> features;
    std::vector<Eigen::VectorXf> rows;
    long num_columns, num_rows;
    
public:
    
    Data() = default; //so we can have a static Data attribute
    
    // Assignment operator
    Data& operator=(const Eigen::MatrixXf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        for (long i = 0; i < this->num_columns; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXf(this->num_rows);
            for (long j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
        }
        this->rows.resize(this->num_rows);
        
        for (long i = 0; i < num_rows; i++)
        {
            this->rows[i] = data.row(i);
        }
        
        return *this;
    }
    
    bool operator==( Data& other)
    {
        return this->data == other.data;
    }
    
    const Eigen::VectorXf& operator[] (int i){return rows[i];}
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return features[i];
    }
    long numRows() const {return num_rows;}
    long numCols() const {return num_columns;}

    friend std::ostream& operator<<(std::ostream& os, const Data& matrix)
    {
        return (os << matrix.data);
    }
};

float MSE(const Eigen::VectorXf& actual)
{
    return actual.squaredNorm() / actual.size();
}

float MSE(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    return (actual - predicted).squaredNorm() / actual.size();
}

Eigen::AutoDiffScalar<Eigen::VectorXf> MSE(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& actual)
{
    return actual.squaredNorm() / actual.size();
}

float loss_func(const Eigen::VectorXf& actual)
{
    return (1.0f/(1.0f+MSE(actual)));
}

float loss_func(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    return (1.0f/(1.0f+MSE(actual, predicted)));
}

namespace InvertedPendulum
{
    struct State
    {
        float x;   // Position
        float v;   // Velocity
    };

    // Constants
    constexpr float g = 9.81f; //gravitational acceleration on earth
    constexpr float l = 4.0f;      //Fixing the length of the pendulum for simplicity and to avoid potential problems with "apparent stability"
    constexpr float T = 100.0f;       // Final time
    constexpr float dt = 0.01f;      // Time step
    constexpr float x_0 = std::numbers::pi_v<float> - 0.14159f;  // Initial angle
    constexpr float v_0 = 0.0f;      // Initial angular velocity
    constexpr float dt_over_6 = dt/6.0f;
    
    int steps = static_cast<int>(T / dt);

//    l \ddot{\theta} + \ddot{x}_p(t)\cos(\theta)  + g \sin (\theta)  = 0
//     => \ddot{\theta} = (-\ddot{x}_p(t)\cos(\theta) -g \sin (\theta))/l
    float acceleration(float theta, float x_p_ddot)
    {
        return (-x_p_ddot*cos(theta) -g*sin(theta))/l;
    }

    // Define the right-hand side of the ODE (returns dx/dt and dv/dt)
    State derivative(const State& state, float x_p_ddot)
    {
        State deriv;
        deriv.x = state.v;                  // dx/dt = v
        deriv.v = acceleration(state.x, x_p_ddot);
        return deriv;
    }
    
    // RK4 step for updating state
    void rk4_step(State& state, float x_p_ddot, float dt)
    {
        State k1 = derivative(state, x_p_ddot);
        State k2 = derivative({state.x + 0.5f * dt * k1.x, state.v + 0.5f * dt * k1.v}, x_p_ddot);
        State k3 = derivative({state.x + 0.5f * dt * k2.x, state.v + 0.5f * dt * k2.v}, x_p_ddot);
        State k4 = derivative({state.x + dt * k3.x, state.v + dt * k3.v}, x_p_ddot);

        state.x = state.x + dt_over_6 * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x);
        state.v = state.v + dt_over_6 * (k1.v + 2.0f * k2.v + 2.0f * k3.v + k4.v);
    }
}

struct Board
{
    static boost::concurrent_flat_map<std::string, Eigen::VectorXf> inline expression_dict;
    static std::atomic<float> inline fit_time = 0.0;
    
    static constexpr float K = 0.0884956f;
    static constexpr float phi_1 = 2.8f;
    static constexpr float phi_2 = 1.3f;
    static int inline __num_features;
    //TODO: Maybe change to unordered_set
    static std::vector<std::string> inline __input_vars;
    static std::vector<std::string> inline __unary_operators;
    static std::vector<std::string> inline __binary_operators;
    static std::vector<std::string> inline __operators;
    static std::vector<std::string> inline __other_tokens;
    static std::vector<std::string> inline __tokens;
    Eigen::VectorXf params; //store the parameters of the expression of the current episode after it's completed
    static Data inline data;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> vel_dist, pos_dist;
    
    static int inline action_size;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    static std::unordered_map<std::string, std::pair<std::string, std::string>> inline feature_mins_maxes;
    
    size_t reserve_amount;
    int num_fit_iter;
    float MSE_curr;
    std::string fit_method;
    std::string fit_grad_method;
    
    bool cache;
    bool const_token;
    bool expression_evaluator_valid_flag = true;
    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0, idx = 0;
    static std::unordered_map<bool, std::unordered_map<bool, std::unordered_map<bool, std::vector<std::string>>>> inline una_bin_leaf_legal_moves_dict;
    std::vector<int> simplify_grasp;
    
    int n; //depth of RPN/PN tree
    std::string expression_type, expression_string;
    static std::mutex inline thread_locker; //static because it needs to protect static members
    std::vector<std::string> pieces; // Create the empty expression list.
    std::vector<std::string> derivat;// Vector to store the derivative.
    std::vector<std::string> derivat2;// Vector to store the second derivative of pieces
    bool visualize_exploration, is_primary;
    std::vector<std::string> (*diffeq)(Board&); //differential equation we want to solve
    float isConstTol;
    float upperVar;
    bool simplify_original;
    
    Board(bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false, bool simplifyOriginal = true) : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, num_fit_iter{numFitIter}, fit_method{fitMethod}, fit_grad_method{fitGradMethod}, is_primary{primary}, simplify_original{simplifyOriginal}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        
        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->visualize_exploration = visualize_exploration;
        this->reserve_amount = 2*std::pow(2,this->n)-1;
        this->pieces.reserve(this->reserve_amount);
        this->cache = cache;
        this->isConstTol = isConstTol;
        this->const_token = const_token;
        this->upperVar = upperVar;
        
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
                Board::__num_features = Board::data[0].size();
                printf("Number of features = %d\n", Board::__num_features);
                Board::__input_vars.clear();
                Board::expression_dict.clear();
                Board::__input_vars.reserve(Board::__num_features);
                for (auto i = 0; i < Board::__num_features; i++)
                {
                    Board::__input_vars.push_back("x"+std::to_string(i));
                }
                Board::__unary_operators = {"~", "log", "ln", "exp", "cos", "sin", "sqrt", "asin", "arcsin", "acos", "arccos", "tanh", "sech"};
                Board::__binary_operators = {"+", "-", "*", "/", "^"};
                Board::__operators.clear();
                for (std::string& i: Board::__unary_operators)
                {
                    Board::__operators.push_back(i);
                }
                for (std::string& i: Board::__binary_operators)
                {
                    Board::__operators.push_back(i);
                }
                Board::__other_tokens = {"0", "1", "2", "4"};
                //Add points at boundary to Board::__other_tokens
                for (const std::string& i: Board::__input_vars)
                {
                    std::string minCoeff_i = std::to_string(Board::data[i].minCoeff());
                    std::string maxCoeff_i = std::to_string(Board::data[i].maxCoeff());
                    Board::__other_tokens.push_back(minCoeff_i); //add smallest element
                    Board::__other_tokens.push_back(maxCoeff_i); //add largest element
                    feature_mins_maxes[i] = std::make_pair(minCoeff_i, maxCoeff_i);
                }
                if (const_token)
                {
                    Board::__other_tokens.push_back("const");
                }
                Board::__tokens = Board::__operators;
                
                for (auto& i: this->Board::__input_vars)
                {
                    Board::__tokens.push_back(i);
                }
                for (auto& i: Board::__other_tokens)
                {
                    Board::__tokens.push_back(i);
                }
                if (const_token)
                {
                    assert(Board::__tokens.back() == "const");
                }
                Board::action_size = Board::__tokens.size();
                
                Board::una_bin_leaf_legal_moves_dict.clear();
                if (const_tokens)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens;
                }
                
                else
                {
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__operators;
                }
                
                Board::una_bin_leaf_legal_moves_dict[true][true][false] = Board::__operators;
                Board::una_bin_leaf_legal_moves_dict[true][false][true] = Board::__unary_operators; //1
                Board::una_bin_leaf_legal_moves_dict[true][false][false] = Board::__unary_operators;
                Board::una_bin_leaf_legal_moves_dict[false][true][true] = Board::__binary_operators; //2
                Board::una_bin_leaf_legal_moves_dict[false][true][false] = Board::__binary_operators;
                
                for (const std::string &i: Board::__input_vars)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                    if (!const_tokens)
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][true][true].push_back(i);
                    }
                }
                if (const_tokens)
                {
                    for (const std::string &i: Board::__other_tokens)
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                        Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                        Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                    }
                }
            });
            
        }
    }
    
    std::string operator[](size_t index) const
    {
        if (index < Board::__tokens.size())
        {
            return Board::__tokens[index];
        }
        throw std::out_of_range("Index out of range");
    }
    
    int __num_binary_ops() const
    {
        int count = 0;
        for (const std::string& token : pieces)
        {
            if (std::find(Board::__binary_operators.begin(), Board::__binary_operators.end(), token) != Board::__binary_operators.end())
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_unary_ops() const
    {
        int count = 0;
        for (const std::string& token : pieces)
        {
            if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end())
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_leaves() const
    {
        int count = 0;
        
        for (const std::string& token : pieces)
        {
            if (!is_unary(token) && !is_binary(token))
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_consts() const
    {
        if (!this->const_token)
        {
            return 0;
        }
        int count = 0;
        
        for (const std::string& token : pieces)
        {
            if (token.substr(0,5) == "const")
            {
                count++;
            }
        }
        return count;
    }
    
    bool is_unary(const std::string& token) const
    {
        return (std::find(__unary_operators.begin(), __unary_operators.end(), token) != __unary_operators.end());
    }
    
    bool is_binary(const std::string& token) const
    {
        return (std::find(__binary_operators.begin(), __binary_operators.end(), token) != __binary_operators.end());
    }
    
    bool is_operator(const std::string& token) const
    {
        return (is_binary(token) || is_unary(token));
    }
    
    bool is_const(const std::string& token) const
    {
        return ((!is_unary(token)) && (!is_binary(token)));
    }
    
    std::string simplifyString(const std::string& x)
    {
        if ((x.size() == 2) && (x[0] == '-') && (x[1] == '0')) //"-0" -> "0"
        {
            return "0";
        }
        unsigned long jdx = x.find(".");
        if (jdx == std::string::npos) //if there's no '.' in x
        {
            return x;
        }
        for (unsigned long i = jdx + 1; i < x.size(); i++) //checking if all the characters after the decimal in x are 0; if not, then return x
        {
            if (x[i] != '0')
            {
                return x;
            }
        }
        std::string temp = x.substr(0, jdx);
        if ((temp.size() == 2) && (temp[0] == '-') && (temp[1] == '0')) //"-0.0000" (repeating) -> "0"
        {
            return "0";
        }
        
        return temp; //"x.0000000" (repeating) -> "x"
    }
    
    void graspSimplifyPrefixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            setPrefixGR(expression, grasp);
        }
    //    print_container(expression, low, up);
        if (expression[low] == "+" || expression[low] == "-")
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]);
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true);
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true);
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if (new_expression[first_arg_idx_high] == "0") //+/- x 0 -> x
            {
                //puts("hi 177");
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //remove +/- operator at beginning
            }
            
            else if (new_expression[first_arg_idx_low] == "0")
            {
                if (expression[low] == "+") //+ 0 y -> y
                {
                    //puts("hi 176");
                    new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + first_arg_idx_high); //remove '+' and '0'
                }
                else //- 0 y -> ~ y
                {
                    //puts("hi 184");
                    new_expression[op_idx] = "~";
                    new_expression.erase(new_expression.begin() + first_arg_idx_low); //'0'
                }
            }
            
            else if ((expression[low] == "-") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression)))
            {
                //puts("hi 221");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "0"; //change "-" to "0";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }
        }
        else if (expression[low] == "*") //* x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); //*
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); //* x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); //* x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if (new_expression[first_arg_idx_high] == "0") //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
            {
                //puts("hi 239");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "0") //* 0 x -> 0
            {
                //puts("hi 245");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else
        {
            for (int i = low; i <= up; i++)
            {
                assert(i < expression.size() && i >= 0);
                new_expression.push_back(expression[i]);
            }
        }
    }

    void graspSimplifyPrefix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
    {
        std::vector<std::string> new_expression;
        new_expression.reserve(expression.size());
        graspSimplifyPrefixHelper(expression, low, up, grasp, new_expression, false);
        expression = new_expression;
    }
    
    void simplifyPN_Helper(std::vector<std::string>& expression)
    {
        bool simplified = true;
        bool isFloat1, isFloat2;
        while (simplified)
        {
            simplified = false;
            if (expression.size() > 1)
            {
                for (size_t i = 0; i < expression.size() - 1; i++)
                {
                    if (is_binary(expression[i]))
                    {
                        isFloat1 = isFloat(expression[i+1]);
                        isFloat2 = isFloat(expression[i+2]);
                        
                        if (isFloat1 && isFloat2)
                        {
                            if (expression[i] == "+")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) + std::stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "-")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) - std::stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "*")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) * std::stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "/")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) / std::stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "^")
                            {
                                expression[i] = simplifyString(std::to_string(std::powf(std::stof(expression[i+1]), std::stof(expression[i+2]))));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "-")
                        {
                            if ((is_const(expression[i+1]) && is_const(expression[i+2])) && (expression[i+1] == expression[i+2])) //- x x => 0
                            {
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "0") //- 0 x -> ~ x
                            {
                                expression[i] = "~";
                                expression.erase(expression.begin() + i + 1);
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && is_const(expression[i+1])) //- x 0 -> x
                            {
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "*")
                        {
                            if (expression[i+1] == "0" && is_const(expression[i+2])) //* 0 x -> 0
                            {
                                //puts("hi 131");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && is_const(expression[i+1])) //* x 0 -> 0
                            {
                                //puts("hi 139");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "1" && is_const(expression[i+2])) //* 1 x -> x
                            {
                                //puts("hi 147");
                                expression[i] = expression[i+2];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && is_const(expression[i+1])) //* x 1 -> x
                            {
                                //puts("hi 155");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "+")
                        {
                            if (expression[i+1] == "0" && is_const(expression[i+2])) //+ 0 x -> x
                            {
                                //puts("hi 167");
                                expression[i] = expression[i+2];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && is_const(expression[i+1])) //+ x 0 -> x
                            {
                                //puts("hi 175");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "/")
                        {
                            if (expression[i+1] == "0" && is_const(expression[i+2])) // / 0 x -> 0
                            {
                                //puts("hi 187");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && is_const(expression[i+1])) // / x 1 -> x
                            {
                                //puts("hi 195");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (is_const(expression[i+1]) && is_const(expression[i+2]) && (expression[i+1] == expression[i+2])) // / x x -> 1
                            {
                                //puts("hi 203");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "^")
                        {
                            if (expression[i+1] == "0" && is_const(expression[i+2])) // ^ 0 x -> 0
                            {
                                //puts("hi 215");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && is_const(expression[i+1])) // ^ x 0 -> 1
                            {
                                //puts("hi 223");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "1" && is_const(expression[i+2])) // ^ 1 x -> 1
                            {
                                //puts("hi 231");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && is_const(expression[i+1])) // ^ x 1 -> x
                            {
                                //puts("hi 239");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        
                    }
                    
                    else if (is_unary(expression[i]) && isFloat(expression[i+1]))
                    {
                        if (expression[i] == "cos")
                        {
                            expression[i] = simplifyString(std::to_string(cos(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "~")
                        {
                            expression[i] = simplifyString(std::to_string(-(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin")
                        {
                            expression[i] = simplifyString(std::to_string(sin(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if ((expression[i] == "ln") || (expression[i] == "log"))
                        {
                            expression[i] = simplifyString(std::to_string(log(std::stof(expression[i+1])))); // Natural log (ln)
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "asin" || expression[i] == "arcsin")
                        {
                            expression[i] = simplifyString(std::to_string(asin(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "acos" || expression[i] == "arccos")
                        {
                            expression[i] = simplifyString(std::to_string(acos(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp")
                        {
                            expression[i] = simplifyString(std::to_string(exp(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sech")
                        {
                            expression[i] = simplifyString(std::to_string(1 / cosh(std::stof(expression[i+1])))); // sech(x) = 1 / cosh(x)
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "tanh")
                        {
                            expression[i] = simplifyString(std::to_string(tanh(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sqrt")
                        {
                            expression[i] = simplifyString(std::to_string(sqrt(std::stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (is_unary(expression[i]))
                    {
                        if (expression[i] == "~" && expression[i+1] == "~")
                        {
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                        {
                            //puts("hi 361");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                        {
                            //puts("hi 369");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "cos" && (expression[i+1] == "acos" || expression[i+1] == "arccos"))
                        {
                            //puts("hi 403");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                        {
                            //puts("hi 411");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin" && (expression[i+1] == "asin" || expression[i+1] == "arcsin"))
                        {
                            //puts("hi 419");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                        {
                            //puts("hi 427");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    void simplifyPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPrefix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
    }
    
    
    void graspSimplifyPostfixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            setPostfixGR(expression, grasp);
        }
    //    print_container(expression, low, up);
        if (expression[up] == "+" || expression[up] == "-")
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true);
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true);
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if (new_expression.back() == "0") // x 0 +/- -> x
            {
                //puts("hi 181");
                new_expression.pop_back();
            }
            
            else if (new_expression[first_arg_idx_high - 1] == "0")
            {
                //puts("hi 184");
                //erase elements from new_expression[first_arg_idx_low] to new_expression[first_arg_idx_high-1] inclusive
                new_expression.erase(new_expression.begin() + first_arg_idx_low, new_expression.begin() + first_arg_idx_high); //0 y + -> y
                if (expression[up] == "-")
                {
                    //puts("hi 187");
                    new_expression.push_back("~"); //0 y - -> y ~
                }
            }
            
            else if ((expression[up] == "-") && ((step = (first_arg_idx_high - first_arg_idx_low)) == (second_arg_idx_high - first_arg_idx_high)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression)))
            {
                //puts("hi 215");
                new_expression[first_arg_idx_low] = "0"; //change first symbol of x' to 0
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.begin() + second_arg_idx_high); //erase the rest of x' and y'
            }
            
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "*") //x y *
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            
            if (new_expression.back() == "0") // x 0 * -> 0 (because, since postfix operators come at the end, if the end of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 *, which is 0)
            {
                //puts("hi 235");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression[first_arg_idx_high - 1] == "0") //0 x * -> 0 (because, since postfix operators come at the end, if the end of the first argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x *, which is 0)
            {
                //puts("hi 241");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else
        {
            for (int i = low; i <= up; i++)
            {
                assert(i < expression.size() && i >= 0);
                new_expression.push_back(expression[i]);
            }
        }
    }
    
    void graspSimplifyPostfix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
    {
        std::vector<std::string> new_expression;
        new_expression.reserve(expression.size());
        graspSimplifyPostfixHelper(expression, low, up, grasp, new_expression, false);
        expression = new_expression;
    }

    void simplifyRPN_Helper(std::vector<std::string>& expression)
    {
        bool simplified = true;
        bool isFloat1, isFloat2;
        while (simplified)
        {
            simplified = false;
            if (expression.size() > 1)
            {
                for (size_t i = 1; i < expression.size(); i++)
                {
                    if (is_binary(expression[i]))
                    {
                        isFloat1 = isFloat(expression[i-1]);
                        isFloat2 = isFloat(expression[i-2]);
                        
                        if (isFloat1 && isFloat2)
                        {
                            if (expression[i] == "+")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) + std::stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "-")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) - std::stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "*")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) * std::stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "/")
                            {
                                expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) / std::stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "^")
                            {
                                expression[i] = simplifyString(std::to_string(std::powf(std::stof(expression[i-2]), std::stof(expression[i-1]))));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "-")
                        {
                            if ((is_const(expression[i-1]) && is_const(expression[i-2])) && (expression[i-1] == expression[i-2])) //x x - => 0
                            {
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "0" && is_const(expression[i-1])) //"0 x -" -> "x ~"
                            {
                                expression[i] = "~";
                                expression.erase(expression.begin() + i - 2);
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0") //"x 0 -" -> "x"
                            {
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "*")
                        {
                            if (expression[i-2] == "0" && is_const(expression[i-1])) //"0 x *" -> "0"
                            {
                                //puts("hi 131");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0" && is_const(expression[i-2])) //"x 0 *" -> "0"
                            {
                                //puts("hi 139");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "1" && is_const(expression[i-1])) //"1 x *" -> "x"
                            {
                                //puts("hi 147");
                                expression[i] = expression[i-1];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && is_const(expression[i-2])) //"x 1 *" -> "x"
                            {
                                //puts("hi 155");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "+")
                        {
                            if (expression[i-2] == "0" && is_const(expression[i-1])) //"0 x +" -> "x"
                            {
                                //puts("hi 167");
                                expression[i] = expression[i-1];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0" && is_const(expression[i-2])) //"x 0 +" -> "x"
                            {
                                //puts("hi 175");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "/")
                        {
                            if (expression[i-2] == "0" && is_const(expression[i-1])) // "0 x /" -> "0"
                            {
                                //puts("hi 187");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && is_const(expression[i-2])) // "x 1 /" -> "x"
                            {
                                //puts("hi 195");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (is_const(expression[i-1]) && is_const(expression[i-2]) && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                            {
                                //puts("hi 203");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                        
                        else if (expression[i] == "^")
                        {
                            if (expression[i-2] == "0" && is_const(expression[i-1])) // "0 x ^" -> "0"
                            {
                                //puts("hi 215");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0" && is_const(expression[i-2])) // "x 0 ^" -> "1"
                            {
                                //puts("hi 223");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "1" && is_const(expression[i-1])) // "1 x ^" -> "1"
                            {
                                //puts("hi 231");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && is_const(expression[i-2])) // "x 1 ^" -> "x"
                            {
                                //puts("hi 239");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                    }
                    
                    else if (is_unary(expression[i]) && isFloat(expression[i-1]))
                    {
                        if (expression[i] == "cos")
                        {
                            expression[i] = simplifyString(std::to_string(cos(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "~")
                        {
                            expression[i] = simplifyString(std::to_string(-(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin")
                        {
                            expression[i] = simplifyString(std::to_string(sin(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if ((expression[i] == "ln") || (expression[i] == "log"))
                        {
                            expression[i] = simplifyString(std::to_string(log(std::stof(expression[i-1])))); // Natural log (ln)
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "asin" || expression[i] == "arcsin")
                        {
                            expression[i] = simplifyString(std::to_string(asin(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "acos" || expression[i] == "arccos")
                        {
                            expression[i] = simplifyString(std::to_string(acos(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp")
                        {
                            expression[i] = simplifyString(std::to_string(exp(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sech")
                        {
                            expression[i] = simplifyString(std::to_string(1 / cosh(std::stof(expression[i-1])))); // sech(x) = 1 / cosh(x)
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "tanh")
                        {
                            expression[i] = simplifyString(std::to_string(tanh(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sqrt")
                        {
                            expression[i] = simplifyString(std::to_string(sqrt(std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (is_unary(expression[i]))
                    {
                        if (expression[i] == "~" && expression[i-1] == "~")
                        {
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                        {
                            //puts("hi 360");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                        {
                            //puts("hi 368");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "cos" && (expression[i-1] == "acos" || expression[i-1] == "arccos"))
                        {
                            //puts("hi 408");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                        {
                            //puts("hi 416");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin" && (expression[i-1] == "asin" || expression[i-1] == "arcsin"))
                        {
                            //puts("hi 424");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                        {
                            //puts("hi 432");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    void simplifyRPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyRPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPostfix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyRPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77180279
     */
    std::pair<int, bool> getPNdepth(const std::vector<std::string>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool binary = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }
        
        if (stop == 0)
        {
            stop = expression.size();
        }
        
        if (!cache)
        {
            this->stack.clear();
            this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            for (size_t i = start; i < stop; i++)
            {
                if (is_binary(expression[i]))
                {
                    this->stack.push_back(2);  // Number of operands
                    this->num_binary++;
                }
                else if (is_unary(expression[i]))
                {
                    this->stack.push_back(1);
                }
                else
                {
                    this->num_leaves++;
                    while (!this->stack.empty() && this->stack.back() == 1) //so the this->stack will shrink one by one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack.pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack.empty())
                    {
                        this->stack.back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth = std::max(this->depth, static_cast<int>(this->stack.size()) + 1);
            }
        }
        else //optimize with caching
        {
            if (not modify) //get_legal_moves()
            {
                if (binary) //Gives the this->depth and completeness of the current PN expression + a binary operator
                {
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.size()) + 2) - 1, this->num_leaves == this->num_binary + 2);
                }
                else if (unary) //Gives the this->depth and completeness of the current PN expression + a unary operator
                {
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.size()) + 2) - 1, this->num_leaves == this->num_binary + 1);
                }
                else if (leaf) //Gives the this->depth and completeness of the current PN expression + a leaf node
                {
                    auto last_filled_op_it = std::find_if(this->stack.rbegin(), this->stack.rend(), [](int i){return i != 1;}); //Find the first element from the back that's not 1
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.rend() - last_filled_op_it) /* this->stack.size() */ + 1) - 1, this->num_leaves == this->num_binary);
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx]))
                {
                    this->stack.push_back(2);  // Number of operands
                    this->num_binary++;
                }
                else if (is_unary(expression[this->idx]))
                {
                    this->stack.push_back(1);
                }
                else
                {
                    this->num_leaves++;
                    while (!this->stack.empty() && this->stack.back() == 1) //so the this->stack will shrink one-by-one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack.pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack.empty())
                    {
                        this->stack.back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth = std::max(this->depth, static_cast<int>(this->stack.size()) + 1);
                this->idx++;
            }
        }
        return std::make_pair(this->depth - 1, this->num_leaves == this->num_binary + 1);
    }
    
    float loss_func()
    {
        using namespace InvertedPendulum;
        // Initial conditions
        State state;
        state.x = x_0;    // x(0) = x0
        state.v = v_0;    // v(0) = 0
        float max_dev = 0.0f, temp = 0.0f, t = 0.0f;
        
        // Time evolution loop
        
        for (int step = 0; step < steps; ++step)
        {
            t = step * dt;
            // Update the state using RK4
            rk4_step(state, expression_evaluator(this->params, this->derivat2, t), dt);
            if (!expression_evaluator_valid_flag)
            {
                this->MSE_curr = FLT_MAX;
                return 0.0f;
            }
            temp = std::fabs(state.x - x_0);
            if (temp > max_dev)
            {
                max_dev = temp;
            }
        }
        
        if (isInvalid(max_dev))
        {
            this->MSE_curr = FLT_MAX;
            return 0.0f;
        }
        
        this->MSE_curr = max_dev;
        
        return (1.0f / (1.0f + this->MSE_curr));
    }
    
    float loss_func(const Eigen::VectorXf& params)
    {
        using namespace InvertedPendulum;
        // Initial conditions
        State state;
        state.x = x_0;    // x(0) = x0
        state.v = v_0;    // v(0) = 0
        float max_dev = 0.0f, temp = 0.0f, t = 0.0f;
        
        // Time evolution loop
        
        for (int step = 0; step < steps; ++step)
        {
            t = step * dt;
            // Update the state using RK4
            rk4_step(state, expression_evaluator(params, this->derivat2, t), dt);
            if (!expression_evaluator_valid_flag)
            {
                this->MSE_curr = FLT_MAX;
                return 0.0f;
            }
            temp = std::fabs(state.x - x_0);
            if (temp > max_dev)
            {
                max_dev = temp;
            }
        }
        
        if (isInvalid(max_dev))
        {
            this->MSE_curr = FLT_MAX;
            return 0.0f;
        }
        
        this->MSE_curr = max_dev;
        
        return (1.0f / (1.0f + this->MSE_curr));
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77128902
     */
    std::pair<int, bool> getRPNdepth(const std::vector<std::string>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }
        
        if (stop == 0)
        {
            stop = expression.size();
        }
        
        if (!cache)
        {
            this->stack.clear();
            bool complete = true;
            
            for (size_t i = start; i < stop; i++)
            {
                if (is_unary(expression[i]))
                {
                    this->stack.back() += 1;
                }
                else if (is_binary(expression[i]))
                {
                    int op2 = this->stack.back();
                    this->stack.pop_back();
                    int op1 = this->stack.back();
                    this->stack.pop_back();
                    this->stack.push_back(std::max(op1, op2) + 1);
                }
                else //leaf
                {
                    this->stack.push_back(1);
                }
            }
            
            while (this->stack.size() > 1)
            {
                int op2 = this->stack.back();
                this->stack.pop_back();
                int op1 = this->stack.back();
                this->stack.pop_back();
                this->stack.push_back(std::max(op1, op2) + 1);
                complete = false;
            }
            
            /*
             e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
             {1, 2, 3, 4, 5}
             {1, 2, 3, 6}
             {1, 2, 7}
             {1, 8}
             {9}
             */
            
            return std::make_pair(this->stack.back() - 1, complete);
        }
        else //optimize with caching
        {
            if (not modify)  //get_legal_moves()
            {
                if (unary) //Gives the this->depth and completeness of the current RPN expression + a unary operator
                {
                    if (this->stack.size() == 1)
                    {
                        return std::make_pair(this->stack.back(), true);
                    }
                    else
                    {
                        int curr_max = std::max(this->stack.back()+1, *(this->stack.end()-2))+1;
                        for (int i = this->stack.size() - 2; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[i-1])+1;
                        }
                        
                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */
                        
                        return std::make_pair(curr_max - 1, false);
                    }
                }
                else if (leaf) //Gives the this->depth and completeness of the current RPN expression + a leaf node
                {
                    if (this->stack.empty())
                    {
                        return std::make_pair(0, true);
                    }
                    else
                    {
                        int curr_max = std::max(this->stack.back(), 1)+1;
                        for (int i = this->stack.size() - 1; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[i-1])+1;
                        }
                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */
                        
                        return std::make_pair(curr_max - 1, false);
                    }
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx]))
                {
                    int op2 = this->stack.back();
                    this->stack.pop_back();
                    int op1 = this->stack.back();
                    this->stack.pop_back();
                    this->stack.push_back(std::max(op1, op2) + 1);
                }
                else if (is_unary(expression[this->idx]))
                {
                    this->stack.back() += 1;
                }
                else //leaf
                {
                    this->stack.push_back(1);
                }
                
                this->idx++;
                if (this->stack.size() == 1)
                {
                    return std::make_pair(this->stack.back() - 1, true);
                }
                
                else
                {
                    int curr_max = std::max(this->stack.back(), *(this->stack.end()-2))+1;
                    for (int i = this->stack.size() - 2; i >= 1; i--)
                    {
                        curr_max = std::max(curr_max, this->stack[i-1])+1;
                    }
                    return std::make_pair(curr_max - 1, false);
                }
            }
            
            return std::make_pair(this->stack.back() - 1, true);
        }
    }
    
    std::vector<std::string> get_legal_moves()
    {
        if (this->expression_type == "prefix")
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators...
            {
                if (this->n != 0) // if the depth is not 0
                {
                    return Board::__operators;
                }
                else // else it's the leaves
                {
                    return Board::una_bin_leaf_legal_moves_dict[false][false][true];
                }
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            if (this->cache)
            {
                //basic constraints for depth
                bool una_allowed = (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, true /*unary*/, false /*leaf*/).first <= this->n);
                bool bin_allowed = (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*binary*/, false /*unary*/, false /*leaf*/).first <= this->n);
                bool leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, false /*unary*/, true /*leaf*/).first < this->n && (num_leaves == num_binary))));
                std::vector<std::string> legal_moves = Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
                
                //more complicated constraints for simplification
                if (leaf_allowed)
                {
                    size_t pcs_sz = pieces.size();
                    if (pcs_sz >= 2 && pieces[pcs_sz-2] == "/") // "/ x{i}" should not result in "/ x{i} x{i}" as that is 1
                    {
                        for (const std::string& i: Board::__input_vars)
                        {
                            if (pieces.back() == i)
                            {
                                if (legal_moves.size() > 1)
                                {
                                    legal_moves.erase(std::remove(legal_moves.begin(), legal_moves.end(), i), legal_moves.end()); //remove "x{i}" from legal_moves
                                }
                                else //if x{i} is the only legal move, then we'll change "/" to another binary operator, like "+", "*", or "^"
                                {
                                    std::vector<std::string> sub_bin_ops = {"*", "+", "^"};
                                    std::uniform_int_distribution<int> distribution(0, 2);
                                    pieces[pcs_sz-2] = sub_bin_ops[distribution(gen)];
                                }
                                break;
                            }
                        }
                    }
                }
                assert(legal_moves.size());
                return legal_moves;
                
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = false, leaf_allowed = false;
                if (Board::__binary_operators.size() > 0)
                {
                    pieces.push_back(Board::__binary_operators[0]);
                    bin_allowed = (getPNdepth(pieces).first <= this->n);
                }
                if (Board::__unary_operators.size() > 0)
                {
                    pieces[pieces.size() - 1] = Board::__unary_operators[0];
                    una_allowed = (getPNdepth(pieces).first <= this->n);
                }
                pieces[pieces.size() - 1] = Board::__input_vars[0];
                leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces).first < this->n && (num_leaves == num_binary))));
                pieces.pop_back();
                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));
                
                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }
        
        else //postfix
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the features and const
            {
                return Board::una_bin_leaf_legal_moves_dict[false][false][true];
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            if (this->cache)
            {
                return Board::una_bin_leaf_legal_moves_dict[((num_leaves >= 1) && (getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*unary*/, false /*leaf*/).first <= this->n))][(num_binary != num_leaves - 1)][(getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*unary*/, true /*leaf*/).first <= this->n)];
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = (num_binary != num_leaves - 1), leaf_allowed = false;
                if (Board::__unary_operators.size() > 0)
                {
                    pieces.push_back(Board::__unary_operators[0]);
                    una_allowed = ((num_leaves >= 1) && (getRPNdepth(pieces).first <= this->n));
                }
                
                pieces[pieces.size() - 1] = Board::__input_vars[0];
                leaf_allowed = (getRPNdepth(pieces).first <= this->n);
                
                pieces.pop_back();
                //                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));
                
                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }
        
    }
    
    //Returns the `expression_type` string form of the expression stored in the vector<std::string> attribute pieces
    std::string expression()
    {
        std::string temp, token;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
//        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            token = pieces[i];
            
            if (token.substr(0,5) == "const")
            {
                temp += ((i!=sz) ? std::to_string((this->params)(std::stoi(token.substr(5)))) + " " : std::to_string((this->params)(std::stoi(token.substr(5)))));
//                if (expression_type == "postfix")
//                {
//                    const_index++;
//                }
//                else
//                {
//                    const_index--;
//                }
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }
    
    std::string _to_infix(bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        std::string result, token;
        
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            
            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) // leaf
            {
                if (token.substr(0,5) == "const")
                {
                    stack.push(std::to_string((this->params)(std::stoi(token.substr(5)))));
                }
                else
                {
                    stack.push(token);
                }
            }
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), pieces[i]) != Board::__unary_operators.end()) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }
        
        return stack.top();
    }
    
    //Returns the `expression_type` string form of the expression stored in the vector<std::string> parameter pieces
    std::string expression(const std::vector<std::string>& pieces)
    {
        std::string temp, token;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
//        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            token = pieces[i];
            
            if (token.substr(0,5) == "const")
            {
                temp += ((i!=sz) ? std::to_string((this->params)(std::stoi(token.substr(5)))) + " " : std::to_string((this->params)(std::stoi(token.substr(5)))));
//                temp += ((i!=sz) ? std::to_string((this->params)(const_index)) + " " : std::to_string((this->params)(const_index)));
//                if (expression_type == "postfix")
//                {
//                    const_index++;
//                }
//                else
//                {
//                    const_index--;
//                }
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }
    
    std::string _to_infix(const std::vector<std::string>& pieces, bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        std::string result, token;
        
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            
            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) // leaf
            {
                if (token.substr(0,5) == "const")
                {
                    stack.push(std::to_string((this->params)(std::stoi(token.substr(5)))));
                }
                else
                {
                    stack.push(token);
                }
            }
            
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), pieces[i]) != Board::__unary_operators.end()) // Unary operator
            {
                std::string operand = stack.top();
                assert(!stack.empty());
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                assert(!stack.empty());
                stack.pop();
                std::string left_operand = stack.top();
                assert(!stack.empty());
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }
        assert(!stack.empty());
        return stack.top();
    }
    
    float expression_evaluator(const Eigen::VectorXf& params, const std::vector<std::string>& pieces, float t)
    {
        std::stack<float> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
//            std::cout << "pieces[i] = " << pieces[i] << '\n';
            assert(token.size());
            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) //not an operator, i.e., a leaf
            {
                if (token.substr(0,5) == "const")
                {
                    int temp_idx = std::stoi(token.substr(5));
                    assert(temp_idx < params.size());
                    stack.push(params(temp_idx));
                }
                else if (token == "0")
                {
                    stack.push(0.0f);
                }
                else if (token == "1")
                {
                    stack.push(1.0f);
                }
                else if (token == "2")
                {
                    stack.push(2.0f);
                }
                else if (token == "4")
                {
                    stack.push(4.0f);
                }
                else if (isFloat(token))
                {
                    stack.push(std::stof(token));
                }
                else if (token == "x0")
                {
                    stack.push(t);
                }
                else
                {
                    throw(std::runtime_error("bad token"));
                }
            }
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end()) // Unary operator
            {
                if (token == "cos")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(cos(temp));
                }
                else if (token == "exp")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(exp(temp));
                }
                else if (token == "sqrt")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(sqrt(temp));
                }
                else if (token == "sin")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(sin(temp));
                }
                else if (token == "asin" || token == "arcsin")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(asin(temp));
                }
                else if (token == "log" || token == "ln")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(log(temp));
                }
                else if (token == "tanh")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(tanh(temp));
                }
                else if (token == "sech")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(1.0f/cosh(temp));
                }
                else if (token == "acos" || token == "arccos")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(acos(temp));
                }
                else if (token == "~") //unary minus
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(-temp);
                }
                
                if (isInvalid(stack.top()))
                {
                    expression_evaluator_valid_flag = false;
                    return stack.top();
                }
                else
                {
                    expression_evaluator_valid_flag = true;
                }
            }
            else // binary operator
            {
                float left_operand = stack.top();
                stack.pop();
                float right_operand = stack.top();
                stack.pop();
                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand + left_operand) : (left_operand + right_operand)));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand - left_operand) : (left_operand - right_operand)));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand * left_operand) : (left_operand * right_operand)));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand / left_operand) : (left_operand / right_operand)));
                }
                else if (token == "^")
                {
                    stack.push((expression_type == "postfix") ? (std::pow(right_operand, left_operand)) : (std::pow(left_operand, right_operand)));
                }
                
                if (isInvalid(stack.top()))
                {
                    expression_evaluator_valid_flag = false;
                    return stack.top();
                }
                else
                {
                    expression_evaluator_valid_flag = true;
                }
            }
        }
        if (isInvalid(stack.top()))
        {
            expression_evaluator_valid_flag = false;
        }
        else
        {
            expression_evaluator_valid_flag = true;
        }
        return stack.top();
    }
    
    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params, const std::vector<std::string>& pieces)
    {
//        for (const auto& i: pieces) {std::cout << i << ' ';}puts("");
        assert(pieces.size());
        std::vector<Eigen::VectorXf> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
//            std::cout << "pieces[i] = " << pieces[i] << '\n';
            assert(token.size());
            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) //not an operator, i.e., a leaf
            {
                if (token.substr(0,5) == "const")
                {
                    int temp_idx = std::stoi(token.substr(5));
                    assert(temp_idx < params.size());
                    stack.push_back(Eigen::VectorXf::Ones(Board::data.numRows())*params(temp_idx));
                }
                else if (token == "0")
                {
                    stack.push_back(Eigen::VectorXf::Zero(Board::data.numRows()));
                }
                else if (token == "1")
                {
                    stack.push_back(Eigen::VectorXf::Ones(Board::data.numRows()));
                }
                else if (token == "2")
                {
                    stack.push_back(Eigen::VectorXf::Ones(Board::data.numRows())*2.0f);
                }
                else if (token == "4")
                {
                    stack.push_back(Eigen::VectorXf::Ones(Board::data.numRows())*4.0f);
                }
                else if (isFloat(token))
                {
                    stack.push_back(Eigen::VectorXf::Ones(Board::data.numRows())*std::stof(token));
                }
                else
                {
                    stack.push_back(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end()) // Unary operator
            {
                if (token == "cos")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().cos());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().cos();
                }
                else if (token == "exp")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().exp());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().exp();
                }
                else if (token == "sqrt")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().sqrt());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().sqrt();
                }
                else if (token == "sin")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().sin());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().sin();
                }
                else if (token == "asin" || token == "arcsin")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().asin());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().asin();
                }
                else if (token == "log" || token == "ln")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().log());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().log();
                }
                else if (token == "tanh")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().tanh());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().tanh();
                }
                else if (token == "sech")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(1/temp.array().cosh());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().cosh();
                }
                else if (token == "acos" || token == "arccos")
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(temp.array().acos());
                    
                    (*(stack.end() - 1)).array() = (*(stack.end() - 1)).array().acos();
                }
                else if (token == "~") //unary minus
                {
//                    Eigen::VectorXf temp = stack.back();
//                    stack.pop_back();
//                    stack.push_back(-temp.array());
                    
                    (*(stack.end() - 1)).array() *= -1;
                }
                if (isInvalid(stack.back()))
                {
                    expression_evaluator_valid_flag = false;
                    return stack.back();
                }
                else
                {
                    expression_evaluator_valid_flag = true;
                }
            }
            else // binary operator
            {
    //                Eigen::VectorXf left_operand = stack.back();
    //                stack.pop_back();
    //                Eigen::VectorXf right_operand = stack.back();
    //                stack.pop_back();
                if (token == "+")
                {
                    (*(stack.end() - 2)).array() += stack.back().array();
                    stack.pop_back();  // Remove the now-unused last element
//                    stack.push_back(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    if (expression_type == "postfix")
                    {
                        (*(stack.end() - 2)).array() -= stack.back().array();
                    }
                    else
                    {
                        (*(stack.end() - 2)).array() = stack.back().array() - (*(stack.end() - 2)).array();
                    }
                    stack.pop_back();
//                    stack.push_back(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    (*(stack.end() - 2)).array() *= stack.back().array();
                    stack.pop_back();
//                    stack.push_back(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    if (expression_type == "postfix")
                    {
                        // Divide *(stack.end() - 2) by stack.back() and replace with result
                        (*(stack.end() - 2)).array() /= stack.back().array();
                        
                    }
                    else
                    {
                        // Divide stack.back() by *(stack.end() - 2) and replace with result
                        (*(stack.end() - 2)).array() = stack.back().array() / (*(stack.end() - 2)).array();
                    }
                    stack.pop_back();  // Remove the now-unused back element
//                    stack.push_back(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    if (expression_type == "postfix")
                    {
                        // Divide *(stack.end() - 2) by stack.back() and replace with result
                        (*(stack.end() - 2)).array() = (*(stack.end() - 2)).array().pow(stack.back().array());
                    }
                    else
                    {
                        // Divide stack.back() by *(stack.end() - 2) and replace with result
                        (*(stack.end() - 2)).array() = stack.back().array().pow((*(stack.end() - 2)).array());
                        
                    }
                    stack.pop_back();  // Remove the now-unused second-to-last element
//                    stack.push_back(((expression_type == "postfix") ? (right_operand.array().pow(left_operand.array())) : (left_operand.array().pow(right_operand.array()))));
                }
                
                if (isInvalid(stack.back()))
                {
                    expression_evaluator_valid_flag = false;
                    return stack.back();
                }
                else
                {
                    expression_evaluator_valid_flag = true;
                }
            }
        }
        assert(!stack.empty() && stack.back().size());
        if (isInvalid(stack.back()))
        {
            expression_evaluator_valid_flag = false;
        }
        else
        {
            expression_evaluator_valid_flag = true;
        }
        return stack.back();
    }

    
    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& parameters, const std::vector<std::string>& pieces) const
    {
        std::stack<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            assert(token.size());
            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) // leaf
            {
                if (token.substr(0,5) == "const")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), parameters[std::stoi(token.substr(5))]));
                }
                else if (token == "0")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 0.0f));
                }
                else if (token == "1")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 1.0f));
                }
                else if (token == "2")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 2.0f));
                }
                else if (token == "4")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 4.0f));
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), std::stof(token)));
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end()) // Unary operator
            {
                if (token == "cos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
            }
            else // binary operator
            {
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> left_operand = stack.top();
                stack.pop();
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> right_operand = stack.top();
                stack.pop();
                
                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    stack.push(((expression_type == "postfix") ? ((left_operand.array()*(right_operand.array().log())).exp()) : ((right_operand.array()*(left_operand.array().log())).exp())));
                }
            }
        }
        return stack.top();
    }
    
    bool AsyncPSO()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXf v(this->params.size());
        float rp, rg;
        
        for (long i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }
        
        float swarm_best_score = this->loss_func();
        float fpi = this->loss_func(particle_positions);
        this->MSE_curr = (1.0f/fpi) - 1.0f;
        
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            this->params = particle_positions;
            swarm_best_score = fpi;
            improved = true;
        }
        
        auto UpdateParticle = [&](int i)
        {
            for (int j = 0; j < this->num_fit_iter; j++)
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((this->params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = this->loss_func(particle_positions); //current score
                this->MSE_curr = (1.0f/fpi) - 1.0f;
                
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = this->loss_func(particle_positions); //calculate the score with the new position
                
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
            }
        };
        
        std::vector<std::future<void>> particles;
        particles.reserve(this->params.size());
        for (int i = 0; i < this->params.size(); i++)
        {
            particles.push_back(std::async(std::launch::async | std::launch::deferred, UpdateParticle, i));
        }
        for (auto& i: particles)
        {
            i.get();
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }

    bool PSO()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXf v(this->params.size());
        float rp, rg;
        
        for (long i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }
        
        float swarm_best_score = this->loss_func();
        float fpi = this->loss_func(particle_positions);
        this->MSE_curr = (1.0f/fpi) - 1.0f;
        
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            this->params = particle_positions;
            improved = true;
            swarm_best_score = fpi;
        }
        
        for (int j = 0; j < this->num_fit_iter; j++)
        {
            for (unsigned short i = 0; i < this->params.size(); i++) //number of particles
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((this->params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = this->loss_func(particle_positions); //current score
                this->MSE_curr = (1.0f/fpi) - 1.0f;
                
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = this->loss_func(particle_positions); //calculate the score with the new position
                
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
            }
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    Eigen::AutoDiffScalar<Eigen::VectorXf> grad_func(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& inputs)
    {
        return MSE(expression_evaluator(inputs, this->pieces));
    }
    
    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            if (this->fit_grad_method == "naive_numerical")
            {
                float low_b, temp;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001f;
                    low_b = (1.0f/this->loss_func(x)) - 1.0f;
                    
                    x(i) = temp + 0.00001f;
                    grad(i) = (((1.0f/this->loss_func(x)) - 1.0f) - low_b) / 0.00002f ;
                    x(i) = temp;
                }
            }
            
            else if (this->fit_grad_method == "autodiff")
            {
                size_t sz = x.size();
                std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>> inputs(sz);
                inputs.reserve(sz);
                for (size_t i = 0; i < sz; i++)
                {
                    inputs[i].value() = x(i);
                    inputs[i].derivatives() = Eigen::VectorXf::Unit(sz, i);
                }
                grad = grad_func(inputs).derivatives();
            }
        }
        else if (this->fit_method == "LevenbergMarquardt")
        {
            using namespace InvertedPendulum;
            // Initial conditions
            State state;
            state.x = x_0;    // x(0) = x0
            state.v = v_0;    // v(0) = 0
            float max_dev = 0.0f, temp = 0.0f, t = 0.0f;
            
            // Time evolution loop
            for (int step = 0; step < steps; ++step)
            {
                t = step * dt;
                // Update the state using RK4
                rk4_step(state, expression_evaluator(params, this->derivat2, t), dt);
                if (!expression_evaluator_valid_flag)
                {
                    grad(0) = 1.0f; //...
                    return 0.0f;
                }
                temp = std::fabs(state.x - x_0);
                if (temp > max_dev)
                {
                    max_dev = temp;
                }
            }
            
            // grad(0): Final position error
            if (std::isnan(max_dev) || max_dev > FLT_MAX)
            {
                grad(0) = 1.0f;
            }
            else if (max_dev < -FLT_MAX)
            {
                grad(0) = -1.0f;
            }
            else
            {
                grad(0) = max_dev;
            }
        }
        return 0.0f; // Return 0 for successful execution
    }
    
    bool LBFGS()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSSolver<float, LBFGSpp::LineSearchMoreThuente> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = this->params;
        float mse = (1.0f/this->loss_func()) - 1.0f;
        try
        {
            solver.minimize((*this), eigenVec, fx);
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        
        //        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
            //            printf("mse = %f -> fx = %f\n", mse, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    bool LBFGSB()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSBParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSBSolver<float> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = this->params;
        float mse = (1.0f/this->loss_func()) - 1.0f;
        try
        {
            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -std::numeric_limits<float>::infinity()), Eigen::VectorXf::Constant(eigenVec.size(), std::numeric_limits<float>::infinity()));
            //            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -10.f), Eigen::VectorXf::Constant(eigenVec.size(), 10.f));
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        catch (std::logic_error& e){}
        
        //        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
            //            printf("mse = %f -> fx = %f\n", mse, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    int values() const
    {
        return Board::data.numRows();
    }
    
    int df(Eigen::VectorXf &x, Eigen::MatrixXf &fjac)
    {
        float epsilon, temp;
        epsilon = 1e-5f;
        
        for (int i = 0; i < x.size(); i++)
        {
            //            Eigen::VectorXf xPlus(x);
            //            xPlus(i) += epsilon;
            //
            //            Eigen::VectorXf xMinus(x);
            //            xMinus(i) -= epsilon;
            //            x(i) -= epsilon;
            
            temp = x(i);
            
            x(i) = temp + epsilon;
            Eigen::VectorXf fvecPlus(values());
            operator()(x, fvecPlus);
            
            x(i) = temp - epsilon;
            Eigen::VectorXf fvecMinus(values());
            operator()(x, fvecMinus);
            
            fjac.block(0, i, values(), 1) = std::move((fvecPlus - fvecMinus) / (2.0f * epsilon));
            
            x(i) = temp;
        }
        return 0;
    }
    
    bool LevenbergMarquardt()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::LevenbergMarquardt<decltype(*this), float> lm(*this);
        float score_before = (1.0f/this->loss_func()) - 1.0f;
        lm.parameters.maxfev = this->num_fit_iter;
        //        std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
        //        std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';
        
        lm.minimize(this->params);
        if (((1.0f/this->loss_func()) - 1.0f) < score_before)
        {
            improved = true;
        }
        
        //        std::cout << "Iterations = " << lm.nfev << '\n';
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    float fitFunctionToData()
    {
        float score = 0.0f;
        std::vector<int> grasp;
        grasp.reserve(this->pieces.size());
        
        //Initial tests, if the `pieces` fails these, then we don't have to check so it no need to compute the second derivative for the loss evaluation
        if (isInvalidExpression(this->pieces)) //First, make sure there are no nans or infs in the expression
        {
            this->MSE_curr = FLT_MAX;
            return score;
        }
        
        if (Board::__num_features == 1) //Ignore the trivial solution (1-d functions)
        {
            float var_val = Variance(expression_evaluator(this->params, this->pieces));
            if (!expression_evaluator_valid_flag)
            {
                this->MSE_curr = FLT_MAX;
                return 0.0f;
            }
            if ((var_val < this->isConstTol) || (var_val > this->upperVar))
            {
                this->MSE_curr = FLT_MAX;
                return score;
            }
        }
        else if (Board::__num_features > 1)
        {
            Eigen::VectorXf expression_eval;
            for (const std::string& i: Board::__input_vars)
            {
                if (this->expression_type == "prefix")
                {
                    this->derivePrefix(0, this->pieces.size() - 1, i, this->pieces, grasp);
                }
                else //postfix
                {
                    this->derivePostfix(0, this->pieces.size() - 1, i, this->pieces, grasp);
                }
                float var_val = Variance(expression_evaluator(this->params, this->derivat));
                if (!expression_evaluator_valid_flag)
                {
                    this->MSE_curr = FLT_MAX;
                    return 0.0f;
                }
                if ((var_val < this->isConstTol) || (var_val > this->upperVar)) //Ignore the trivial solution (N-d functions)!
                {
                    this->MSE_curr = FLT_MAX;
                    return score;
                }
            }
        }
        
        //computing the second derivative wrt time and storing that in `this->derivat2` for the rest of this function
        if (this->expression_type == "prefix")
        {
            this->derivePrefix(0, this->pieces.size() - 1, "x0", this->pieces, grasp);
        }
        else //postfix
        {
            this->derivePostfix(0, this->pieces.size() - 1, "x0", this->pieces, grasp);
        }
        this->derivat2 = this->derivat;
        if (this->expression_type == "prefix")
        {
            this->derivePrefix(0, this->derivat2.size() - 1, "x0", this->derivat2, grasp);
        }
        else //postfix
        {
            this->derivePostfix(0, this->derivat2.size() - 1, "x0", this->derivat2, grasp);
        }
        this->derivat2 = this->derivat;
        ((this->expression_type == "prefix") ? simplifyPN(this->derivat2) : simplifyRPN(this->derivat2));
        if (isInvalidExpression(this->derivat2)) //Make sure there are no nans or infs in the second derivative
        {
            this->MSE_curr = FLT_MAX;
            return score;
        }
        
        if (this->params.size())
        {
            bool improved = true;
            if (this->fit_method == "PSO")
            {
                improved = PSO();
            }
            else if (this->fit_method == "AsyncPSO")
            {
                improved = AsyncPSO();
            }
            else if (this->fit_method == "LBFGS")
            {
                improved = LBFGS();
            }
            else if (this->fit_method == "LBFGSB")
            {
                improved = LBFGSB();
            }
            else if (this->fit_method == "LevenbergMarquardt")
            {
                improved = LevenbergMarquardt();
            }
            Eigen::VectorXf temp_vec;
            
            if (improved) //If improved, update the expression_dict with this->params
            {
                if (Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.visit(this->expression_string, [&](auto& x)
                    {
                        x.second = this->params;
                    });
                }
                else
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }
            }
            Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
            {
                temp_vec = x.second;
            });
            
            float var_val = Variance(expression_evaluator(temp_vec, this->pieces));
            if (!expression_evaluator_valid_flag)
            {
                this->MSE_curr = FLT_MAX;
                return 0.0f;
            }
            if ((var_val < this->isConstTol) || (var_val > this->upperVar))
            {
                this->MSE_curr = FLT_MAX;
                return score;
            }
            return this->loss_func();
        }
        else
        {
            return this->loss_func();
        }
        return score;
    }
    
    /*
     Check whether the given player has created a
     complete (depth self.n) expression (again), and
     checks if it is a complete PN/RPN expression.
     Returns the score of the expression if complete,
     where 0 <= score <= 1 and -1 if not complete or if
     the desired depth has not been reached.
     */
    float complete_status(bool cache = true)
    {
        if (this->pieces.empty())
        {
            this->stack.clear();
            this->idx = 0;
            if (this->expression_type == "prefix")
            {
                this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            }
        }
        
        auto [depth, complete] =  ((this->expression_type == "prefix") ? getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) : getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/)); //structured binding :)
        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            if (visualize_exploration)
            {
                //whenever. TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            
            if (is_primary)
            {
//                printf("before: "); for (const auto& i: this->pieces) {std::cout << i << ' ';}puts("");
                if (this->simplify_original)
                {
                    ((this->expression_type == "prefix") ? simplifyPN(this->pieces) : simplifyRPN(this->pieces));
                }
//                printf("after: "); for (const auto& i: this->pieces) {std::cout << i << ' ';}puts("");

                this->expression_string.clear();
                this->expression_string.reserve(8*pieces.size());
                size_t const_count = 0;
                
                for (std::string& token: this->pieces)
                {
                    if (token.substr(0,5) == "const")
                    {
                        token = "const" + std::to_string(const_count++);
                        assert(isFloat(token.substr(5)));
                    }
                    this->expression_string += token+" ";
                }
                
                if (!Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, Eigen::VectorXf());
                }
                
                Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                {
                    this->params = x.second;
                });
                
                if (!this->params.size())
                {
                    this->params.setOnes(this->__num_consts());
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }
                
                return fitFunctionToData();
            }
            return 0.0f;
        }
    }
    const Eigen::VectorXf& operator[] (int i)
    {
        return Board::data[i];
    }
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return Board::data[i];
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Board& b)
    {
        return (os << b.data);
    }
    
    //Function to compute the LGB or RGB, from https://www.jstor.org/stable/43998756
    //(top of pg. 165)
    void GB(size_t z, size_t& ind, const std::vector<std::string>& individual)
    {
        do
        {
            ind = ((expression_type == "prefix") ? ind+1 : ind-1);
            if (is_unary(individual[ind]))
            {
                GB(1, ind, individual);
            }
            else if (is_binary(individual[ind]))
            {
                GB(2, ind, individual);
            }
            --z;
        } while (z);
    }
    
    //Computes the grasp of an arbitrary element pieces[i],
    //from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
    int GR(size_t i, const std::vector<std::string>& individual)
    {
        size_t start = i;
        size_t& ptr_lgb = start;
        if (is_unary(individual[i]))
        {
            GB(1, ptr_lgb, individual);
        }
        else if (is_binary(individual[i]))
        {
            GB(2, ptr_lgb, individual);
        }
        return ((expression_type == "prefix") ? ( ptr_lgb - i) : (i - ptr_lgb));
    }
    
    //Adds pairs containing the starting and stopping indices for each
    //depth-n sub-expression in the expression individual
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::vector<std::string>& individual)
    {
        size_t temp;
        for (size_t k = 0; k < individual.size(); k++)
        {
            temp = k; //we don't want to change k
            size_t& ptr_GB = temp;
            
            if (is_unary(individual[k]))
            {
                GB(1, ptr_GB, individual);
            }
            else if (is_binary(individual[k]))
            {
                GB(2, ptr_GB, individual);
            }
            else if (this->n == 0) //depth-0 sub-trees are leaf-nodes
            {
                sub_exprs.push_back(std::make_pair(k, k));
                continue;
            }
            
            auto [start, stop] = std::make_pair( std::min(k, ptr_GB), std::max(k, ptr_GB));
            //            std::cout << "start, stop = " << start << " , " << stop << '\n';
            auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(individual, start, stop+1, false /*cache*/) : getRPNdepth(individual, start, stop+1));
            
            if (complete && (depth == this->n))
            {
                sub_exprs.push_back(std::make_pair(start, stop));
            }
        }
    }
    
    bool areDerivatRangesEqual(int start_idx_1, int start_idx_2, int num_steps)
    {
        int stop_idx_1 = start_idx_1 + num_steps;
        
        for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
        {
            if (derivat[i] != derivat[j])
            {
                return false;
            }
        }
        return true;
    }
    
    bool areExpressionRangesEqual(int start_idx_1, int start_idx_2, int num_steps, const std::vector<std::string>& expression)
    {
        int stop_idx_1 = start_idx_1 + num_steps;
        
        for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
        {
            if (expression[i] != expression[j])
            {
                return false;
            }
        }
        return true;
    }
    
    void setPrefixGR(const std::vector<std::string>& prefix, std::vector<int>& grasp)
    {
        grasp.reserve(prefix.size());
        for (size_t k = 0; k < prefix.size(); ++k)
        {
            grasp.push_back(GR(k, prefix));
        }
    }
    
    /*
     low and up: lower and upper Index bounds, respectively, for the piece of the array prefix which is to be the subject of the processing.
     dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
     */
    void derivePrefixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(1000);
            //        Index = 0;
            setPrefixGR(prefix, grasp);
        }
        //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
        //Define `grasp` of prefix[i], i.e., the number of elements forming operands of prefix[i] (grasp(operand) = 0)
        //The grasped elements of prefix[i] are the elements forming operands of prefix[i]
        //The left-grasp-bound (LGB) of prefix[i] is the Index of the left-most grasped element of prefix[i] in the array prefix
        //For the expression formed by prefix[i] and its grasped elements, the element prefix[i] is termed the `head` or `main element`
        //ANY element prefix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of prefix[i] in the array prefix (so operands are 0-arity operators)
        //For an N-arity operator prefix[i]:
        //The `grasp` of prefix[i]  is equal to i - LGB(prefix[i])
        //grasp(prefix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(prefix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(prefix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = prefix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
        //If the grasp of an arbitrary prefix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator prefix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (prefix[i-1] and prefix[i-2]) is also an operator (unary or binary).
        //prefix[numElements] is certainly an operator (unary or binary)
        
        //if prefix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to prefix[up-1]
        //then the grasped elements of op2 are the elements from prefix[up-1-grasp[up-1]] to prefix[up-1]
        //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> prefix[up] = "+" is binary
        //so prefix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
        //of prefix[up-1] are the elements [(prefix[up-1-grasp[up-1]] = prefix[6-3] = prefix[3]), prefix[up-1] = prefix[6]]
        //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to prefix[up-grasp(op2)-2] which is equal to prefix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from prefix[low = 0] to prefix[up-2-grasp[up-1]]
        //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->prefix[up] = "+" is binary
        //so prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2] = "*" is the head of the first argument of "+" and so the grasped elements
        //of prefix[up-grasp(op2)-2] are the elements [prefix[low] = prefix[0], prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2]]
        //i.e., the elements {"x", "x", "*"}
        
        if (std::find(prefix.begin(), prefix.end(), dx) == prefix.end())
        {
            this->derivat.push_back("0");
            return;
        }
        
        if (prefix[low] == "+" || prefix[low] == "-")
        {
            int op_idx = this->derivat.size();
            this->derivat.push_back(prefix[low]); //+/-
            int temp = low+1+grasp[low+1];
            int x_prime_low = this->derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true);  /* +/- x' */
            int x_prime_high = this->derivat.size();
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* +/- x' y' */
            int y_prime_high = derivat.size();
            int step;
            
            /*
             Simplification cases:
             
             1.) y' == 0, +/-, x'
             2.) x' == 0,   +, y'
             3.) x' == 0,   -, ~ y'
             
             */
            
            if (derivat[x_prime_high] == "0") //1.) +/- x' 0 -> x'
            {
                //            puts("hi 147");
                //remove y'
                if (x_prime_high == static_cast<int>(derivat.size()) - 1)
                {
                    derivat.pop_back();
                }
                else
                {
                    derivat.erase(derivat.begin() + x_prime_high, derivat.end());
                }
                derivat.erase(derivat.begin() + op_idx); //remove +/- operator at beginning
            }
            
            else if (derivat[x_prime_low] == "0") //2.) and 3.)
            {
                //            puts("hi 162");
                if (prefix[low] == "+") //2.) + 0 y' -> y'
                {
                    derivat.erase(derivat.begin() + op_idx, derivat.begin() + x_prime_high); //remove "+" and "x'"
                }
                else //3.) prefix[low] == "-", - 0 y' -> ~ y'
                {
                    //                puts("hi 170");
                    derivat[op_idx] = "~"; //change binary minus to unary minus
                    derivat.erase(derivat.begin() + x_prime_low); //remove x'
                }
            }
            else if ((prefix[low] == "-") && ((step = (y_prime_high - x_prime_high)) == (x_prime_high - x_prime_low)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
                //                puts("hi 194");
                assert(derivat[op_idx] == prefix[low]);
                derivat[op_idx] = "0"; //change "-" to "0";
                derivat.erase(derivat.begin() + op_idx + 1, derivat.begin() + y_prime_high);
            }
        }
        else if (prefix[low] == "*")
        {
            derivat.push_back("+"); /* +  */
            derivat.push_back("*"); /* + * */
            int x_low = derivat.size();
            int temp = low+1+grasp[low+1];
            for (int k = low+1; k <= temp; k++) /* + * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* 0 y' -> 0
            {
                //            puts("hi 187");
                derivat[x_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_low); //erase x
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* + * x y' */
                if (derivat[y_prime_low] == "0") //* x 0 -> 0
                {
                    //                puts("hi 197");
                    derivat[x_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_low, derivat.end()); //erase x and y'
                }
                else if (derivat[x_low] == "1") //* 1 y' -> y'
                {
                    //                puts("hi 203");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + x_low + 1); //erase "*" and "1"
                }
                else if (derivat[y_prime_low] == "1") //* x 1 -> x
                {
                    //                puts("hi 208");
                    derivat.pop_back(); //remove "1"
                    derivat.erase(derivat.begin() + x_low - 1); //remove "*"
                }
            }
            derivat.push_back("*"); /* + * x y' * */
            int x_prime_low = derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* + * x y' * x' */
            if (derivat[x_prime_low] == "0") //* 0 y -> 0
            {
                //            puts("hi 218");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_prime_low); //erase x'
            }
            else
            {
                int y_low = derivat.size();
                for (int k = temp+1; k <= temp+1+grasp[temp+1]; k++)
                {
                    derivat.push_back(prefix[k]); /* + * x y' * x' y */
                }
                if (derivat[y_low] == "0") //* x' 0 -> 0
                {
                    //                puts("hi 231");
                    derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //erase x' and y
                }
                else if (derivat[x_prime_low] == "1") //* 1 y -> y
                {
                    //                puts("hi 237");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
                else if (derivat[y_low] == "1") //* x' 1 -> x'
                {
                    //                puts("hi 242");
                    derivat.pop_back(); //remove "1"
                    assert(derivat[x_prime_low - 1] == "*");
                    derivat.erase(derivat.begin() + x_prime_low - 1); //remove "*"
                }
            }
            if (derivat[x_low - 1] == "0") //+ 0 * x' y -> * x' y
            {
                //            puts("hi 249");
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //remove "+" and "0"
            }
            else if (derivat[x_prime_low - 1] == "0") //+ * x y' 0 -> * x y'
            {
                //            puts("hi 254");
                assert(static_cast<int>(derivat.size()) == x_prime_low);
                derivat.erase(derivat.begin() + x_low - 2); //erase "+"
                derivat.pop_back(); //remove "0"
            }
        }
        
        else if (prefix[low] == "/")
        {
            int div_idx = derivat.size();
            derivat.push_back("/"); /* / */
            derivat.push_back("-"); /* / - */
            derivat.push_back("*"); /* / - * */
            int temp = low+1+grasp[low+1];
            int x_prime_low = derivat.size();
            int k;
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* / - * x' */
            if (derivat[x_prime_low] == "0") //* 0 y -> 0
            {
                //            puts("hi 297");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                assert(x_prime_low + 1 == static_cast<int>(derivat.size()));
                derivat.pop_back(); //remove x', which is 0
            }
            else
            {
                int y_low = derivat.size();
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y */
                {
                    derivat.push_back(prefix[k]);
                }
                if (derivat[y_low] == "0") //* x' 0 -> 0
                {
                    //                puts("hi 312");
                    derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //remove x' and 0
                }
                else if (derivat[y_low] == "1") //* x' 1 -> x'
                {
                    //                puts("hi 318");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                    derivat.pop_back(); //erase the "1"
                }
                else if (derivat[x_prime_low] == "1") //* 1 y -> y
                {
                    //                    puts("hi 326");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
            }
            derivat.push_back("*"); /* / - * x' y * */
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* / - * x' y * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* 0 y' -> 0
            {
                //            puts("hi 338");
                derivat.erase(derivat.begin() + x_low - 1); //erase "*"
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* / - * x' y * x y' */
                if (derivat[y_prime_low] == "0") //* x 0 -> 0
                {
                    //                puts("hi 347");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and x
                }
                else if (derivat[x_low] == "1") //* 1 y' -> y'
                {
                    //                puts("hi 352");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and 1
                }
                else if (derivat[y_prime_low] == "1") //* x 1 -> x
                {
                    //                puts("hi 357");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1); //erase "*"
                    derivat.pop_back(); //remove the "1"
                }
            }
            
            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - (x_low - 1))) && (areDerivatRangesEqual(x_prime_low - 1, x_low - 1, k))) //- thing1 thing1 -> 0
            {
                //            puts("hi 367");
                derivat[div_idx] = "0";
                derivat.erase(derivat.begin() + div_idx + 1, derivat.end()); //erase everything else
            }
            else
            {
                if (derivat[x_prime_low - 1] == "0") //- 0 * x y' -> ~ * x y'
                {
                    //                puts("hi 375");
                    derivat[x_prime_low - 2] = "~"; //change "-" to "~"
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "0"
                }
                else if (derivat[x_low - 1] == "0") //- * x' y 0 -> * x' y
                {
                    //                    puts("hi 381");
                    assert(static_cast<int>(derivat.size()) == x_low);
                    derivat.erase(derivat.begin() + x_prime_low - 2); //erase the "-"
                    derivat.pop_back(); //erase the "0"
                }
                derivat.push_back("*"); /* / - * x' y * x y' * */
                int y_low = derivat.size();
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y */
                {
                    derivat.push_back(prefix[k]);
                }
                if (derivat[y_low] == "1") // / - * x' y * x y' * 1 1 ->  - * x' y * x y'
                {
                    //                puts("hi 381");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + y_low - 1); //erase "*"
                    derivat.erase(derivat.begin() + div_idx); //erase "/"
                    derivat.pop_back(); //erase "1"
                }
                else
                {
                    for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y y */
                    {
                        derivat.push_back(prefix[k]);
                    }
                }
            }
        }
        
        else if (prefix[low] == "^")
        {
            derivat.push_back("*"); /* * */
            derivat.push_back("^"); /* * ^ */
            int temp = low+1+grasp[low+1];
            int k;
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* * ^ x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* ^ 0 y (* ln 0 y)' -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
                //            puts("hi 454");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            else if (derivat[x_low] == "1") //* ^ 1 y (* ln 1 y)' -> 0 (because ln(1) is 0)
            {
                //            puts("hi 461");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat[x_low] = "0"; //change "1" to "0"
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            int y_low = derivat.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ^ x y */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[y_low] == "0") //* ^ x 0 (* ln x 0)' -> 0
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
                //            puts("hi 474");
                derivat[x_low - 2] = "0"; //change "*" to "0)
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
                return;
            }
            else if (derivat[y_low] == "1") //^ x 1 -> x
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
                derivat.pop_back(); //erase the "1"
                derivat.erase(derivat.begin() + x_low - 1); //erase the "*"
                //            puts("hi 485");
            }
            std::vector<std::string> prefix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> * ln x y
            prefix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            prefix_temp.push_back("*"); /* * */
            prefix_temp.push_back("ln"); /* * ln */
            int x_temp_low = prefix_temp.size();
            for (k = low+1; k <= temp; k++) /* * ln x */
            {
                prefix_temp.push_back(prefix[k]);
            }
            y_low = prefix_temp.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ln x y */
            {
                prefix_temp.push_back(prefix[k]);
            }
            if (prefix_temp[y_low] == "1") //* ln x 1 -> ln x
            {
                //            puts("hi 506");
                assert(y_low == static_cast<int>(prefix_temp.size()) - 1);
                prefix_temp.pop_back(); //remove the "1"
                prefix_temp.erase(prefix_temp.begin() + x_temp_low - 2); //erase the "*"
            }
            setPrefixGR(prefix_temp, grasp_temp);
            int temp_term_low = derivat.size();
            //        derivat.push_back("1");
            derivePrefixHelper(0, prefix_temp.size() - 1, dx, prefix_temp, grasp_temp, true); /* * ^ x y (* ln x y)' */
            if (derivat[temp_term_low] == "0") //* ^ x y 0 -> 0
            {
                //            puts("hi 516");
                derivat[x_low - 2] = "0"; //changing "*" to "0"
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
            }
            else if (derivat[temp_term_low] == "1") //* ^ x y 1 -> ^ x y
            {
                //            puts("hi 522");
                assert(temp_term_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2); //erasing "*"
                derivat.pop_back(); //erasing the "1"
            }
        }
        
        else if (prefix[low] == "cos")
        {
            derivat.push_back("*"); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == "0") //* 0 ~ sin x -> 0
            {
                //                puts("hi 538");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("~"); /* * x' ~ */
            derivat.push_back("sin"); /* * x' ~ sin */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' ~ sin x */
            }
            if (derivat[x_prime_low] == "1") //* 1 ~ sin x -> ~ sin x
            {
                //                puts("hi 551");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (prefix[low] == "sin")
        {
            derivat.push_back("*"); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == "0") //* 0 cos x -> 0
            {
                //                puts("hi 565");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("cos"); /* * x' cos */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' cos x */
            }
            if (derivat[x_prime_low] == "1") //* 1 cos x -> cos x
            {
                //                puts("hi 577");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (prefix[low] == "sqrt")
        {
            derivat.push_back("/");         /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0")
            {
                //            puts("hi 590");
                assert(x_prime_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "/"
                return;
            }
            derivat.push_back("*");         /* / x' * */
            derivat.push_back("2");         /* / x' * 2 */
            derivat.push_back("sqrt");      /* / x' * 2 sqrt */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' * 2 sqrt x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (prefix[low] == "log" || prefix[low] == "ln")
        {
            derivat.push_back("/");               /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0") // / 0 x -> 0
            {
                //                puts("hi 578");
                assert(static_cast<int>(derivat.size()) - 1 == x_prime_low);
                derivat[x_prime_low - 1] = "0"; //change "/" to 0
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
                return;
            }
            int x_low = derivat.size();
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]);      /* / x' x */
            }
            int step = derivat.size() - x_low;
            if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) // / something something -> 1
            {
                //                puts("hi 591");
                derivat[x_prime_low - 1] = "1"; //change "/" to 0
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
            }
        }
        
        else if (prefix[low] == "asin" || prefix[low] == "arcsin")
        {
            derivat.push_back("/");   /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0")
            {
                //                puts("hi 640");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "/"
                return;
            }
            derivat.push_back("sqrt"); /* / x' sqrt */
            derivat.push_back("-");    /* / x' sqrt - */
            derivat.push_back("1");    /* / x' sqrt - 1 */
            derivat.push_back("*");    /* / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (prefix[low] == "acos" || prefix[low] == "arccos")
        {
            derivat.push_back("~");   /* ~ */
            derivat.push_back("/");   /* ~ / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ / x' */
            if (derivat[x_prime_low] == "0")
            {
                //            puts("hi 668");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 2, derivat.begin() + x_prime_low); //erase "~" and "/"
                return;
            }
            derivat.push_back("sqrt"); /* ~ / x' sqrt */
            derivat.push_back("-");    /* ~ / x' sqrt - */
            derivat.push_back("1");    /* ~ / x' sqrt - 1 */
            derivat.push_back("*");    /* ~ / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (prefix[low] == "tanh")
        {
            derivat.push_back("*");      //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0")
            {
                //                puts("hi 696");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //delete the "*"
                return;
            }
            derivat.push_back("*");      //* x' *
            derivat.push_back("sech");   //* x' * sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back("sech");   //* x' * sech x sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x sech x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == "1") //* 1 * sech x sech x -> * sech x sech x
            {
                //                puts("hi 715");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (prefix[low] == "sech")
        {
            derivat.push_back("*"); //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0") //* 0 * ~ sech x tanh x -> 0
            {
                //                puts("hi 722");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "*"
                return;
            }
            derivat.push_back("*");      //* x' *
            derivat.push_back("~");      //* x' * ~
            derivat.push_back("sech");   //* x' * ~ sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back("tanh");   //* x' * ~ sech x tanh
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x tanh x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
            {
                //                puts("hi 742");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (prefix[low] == "exp")
        {
            derivat.push_back("*");               //*
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0") //* 0 exp x -> 0
            {
                //            puts("hi 682");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("exp");           //* x' exp
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]);      //* x' exp x
            }
            if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
            {
                //                puts("hi 694");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (prefix[low] == "~")
        {
            int temp = low+1;
            int un_minus_idx = derivat.size();
            derivat.push_back(prefix[low]); /* ~ */
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ x' */
            if (derivat[x_prime_low] == "~")
            {
                //                puts("hi 590");
                derivat.erase(derivat.begin() + un_minus_idx, derivat.begin() + x_prime_low + 1); //erase the two "~"
            }
        }
        
        else
        {
            if (prefix[low] == dx)
            {
                this->derivat.push_back("1");
            }
            else
            {
                this->derivat.push_back("0");
            }
        }
    }
    
    void derivePrefix(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp)
    {
        derivePrefixHelper(low, up, dx, prefix, grasp, false);
    }
    
    void setPostfixGR(const std::vector<std::string>& postfix, std::vector<int>& grasp)
    {
        grasp.reserve(postfix.size()); //grasp[k] = GR( postfix[k]), k = 1, ... ,i.
        //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
        //if you set k = 0 too.
        for (size_t k = 0; k < postfix.size(); ++k)
        {
            grasp.push_back(GR(k, postfix));
        }
    }
    
    /*
     low and up: lower and upper Index bounds, respectively, for the piece of the array postfix which is to be the subject of the processing.
     dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
     */
    void derivePostfixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(1000);
            //        Index = 0;
            setPostfixGR(postfix, grasp);
        }
        //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
        //Define `grasp` of postfix[i], i.e., the number of elements forming operands of postfix[i] (grasp(operand) = 0)
        //The grasped elements of postfix[i] are the elements forming operands of postfix[i]
        //The left-grasp-bound (LGB) of postfix[i] is the Index of the left-most grasped element of postfix[i] in the array postfix
        //For the expression formed by postfix[i] and its grasped elements, the element postfix[i] is termed the `head` or `main element`
        //ANY element postfix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of postfix[i] in the array postfix (so operands are 0-arity operators)
        //For an N-arity operator postfix[i]:
        //The `grasp` of postfix[i]  is equal to i - LGB(postfix[i])
        //grasp(postfix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(postfix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(postfix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = postfix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
        //If the grasp of an arbitrary postfix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator postfix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (postfix[i-1] and postfix[i-2]) is also an operator (unary or binary).
        //postfix[numElements] is certainly an operator (unary or binary)
        
        //if postfix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to postfix[up-1]
        //then the grasped elements of op2 are the elements from postfix[up-1-grasp[up-1]] to postfix[up-1]
        //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> postfix[up] = "+" is binary
        //so postfix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
        //of postfix[up-1] are the elements [(postfix[up-1-grasp[up-1]] = postfix[6-3] = postfix[3]), postfix[up-1] = postfix[6]]
        //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to postfix[up-grasp(op2)-2] which is equal to postfix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from postfix[low = 0] to postfix[up-2-grasp[up-1]]
        //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->postfix[up] = "+" is binary
        //so postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2] = "*" is the head of the first argument of "+" and so the grasped elements
        //of postfix[up-grasp(op2)-2] are the elements [(postfix[low] = postfix[0], postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2]]
        //i.e., the elements {"x", "x", "*"}
        
        if (std::find(postfix.begin(), postfix.end(), dx) == postfix.end())
        {
            this->derivat.push_back("0");
            return;
        }
        
        if (postfix[up] == "+" || postfix[up] == "-")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true);  /*Putting x'*/
            int x_prime_high = derivat.size();
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /*Putting y'*/
            int y_prime_high = derivat.size();
            int step;
            
            /*
             Simplification cases:
             
             1.) y' == 0, +/-, x'
             2.) x' == 0,   +, y'
             3.) x' == 0,   -, y' ~
             
             */
            
            if (derivat.back() == "0") //1.) x' 0 + -> x'
            {
                //            puts("hi 145");
                derivat.pop_back();
            }
            
            else if (derivat[x_prime_high - 1] == "0")
            {
                //            puts("hi 151");
                //erase elements from derivat[x_prime_low] to derivat[x_prime_high-1] inclusive
                derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + x_prime_high); //0 y + -> y
                if (postfix[up] == "-") //3.)
                {
                    //                puts("hi 156");
                    derivat.push_back("~"); //0 y - -> y ~
                }
            }
            
            else if ((postfix[up] == "-") && ((step = (x_prime_high - x_prime_low)) == (y_prime_high - x_prime_high)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
                //                puts("hi 180");
                derivat[x_prime_low] = "0"; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.begin() + y_prime_high); //erase the rest of x' and y'
            }
            
            else
            {
                derivat.push_back(postfix[up]);
            }
        }
        else if (postfix[up] == "*")
        {
            int x_low = derivat.size();
            for (int k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y' * -> 0
            {
                //            puts("hi 176");
            }
            else
            {
                int x_high = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x y' */
                if (derivat.back() == "0") //x 0 * -> 0
                {
                    //                puts("hi 184");
                    derivat[x_low] = "0"; //change first symbol of x to 0
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase rest of x and y'
                }
                else if (derivat[x_high - 1] == "1") //1 y' * -> y'
                {
                    //                puts("hi 190");
                    assert(x_low == x_high - 1);
                    derivat.erase(derivat.begin() + x_low); //erase the x since it's 1
                }
                else if (derivat.back() == "1") //x 1 * -> x
                {
                    //                puts("hi 196");
                    derivat.pop_back(); //remove the y' since it's 1
                }
                else
                {
                    derivat.push_back("*"); /* x y' "*" */
                }
            }
            
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x y' "*" x' */
            if (derivat.back() == "0") //0 y * -> 0
            {
                //            puts("hi 209");
            }
            else
            {
                int y_low = derivat.size();
                for (int k = up-1-grasp[up-1]; k <= up - 1; k++)
                {
                    derivat.push_back(postfix[k]); /* x y' "*" x' y */
                }
                if (derivat.back() == "0") //x' 0 * -> 0
                {
                    //                puts("hi 220");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + y_low); //erase x'
                }
                else if (derivat[y_low - 1] == "1") //1 y * -> y
                {
                    //                puts("hi 225");
                    assert(y_low - 1 == x_prime_low);
                    derivat.erase(derivat.begin() + x_prime_low); //remove the 1
                }
                else if (derivat.back() == "1") //x' 1 * -> x'
                {
                    //                puts("hi 231");
                    derivat.pop_back(); //remove the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x y' "*" x' y "*" */
                }
            }
            if (derivat[x_prime_low - 1] == "0") // 0 x' y "*" + -> x' y "*"
            {
                //                puts("hi 236");
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase 0
            }
            else if (derivat.back() == "0") //x y' "*" 0 + -> x y' "*"
            {
                //                puts("hi 241");
                derivat.pop_back();
            }
            else
            {
                derivat.push_back("+"); /* x y' "*" x' y "*" + */
            }
        }
        
        else if (postfix[up] == "/")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x' */
            int k;
            if (derivat.back() == "0") //0 y * -> 0
            {
                //            puts("hi 286");
            }
            else
            {
                int y_low = derivat.size();
                for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x' y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "0") //x' 0 * -> 0
                {
                    //                puts("hi 297");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end() - 1); //erase x'
                }
                else if (derivat.back() == "1") //x' 1 * -> x'
                {
                    //                puts("hi 302");
                    derivat.pop_back(); //remove the "1"
                }
                else if (derivat[y_low-1] == "1") //1 y * -> y
                {
                    //                puts("hi 307");
                    derivat.erase(derivat.begin() + y_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x' y *  */
                }
            }
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x' y * x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y' * -> 0
            {
                //            puts("hi 322");
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x' y * x y' */
                if (derivat.back() == "0") //x 0 * -> 0
                {
                    //                puts("hi 330");
                    derivat.erase(derivat.begin() + x_low, derivat.begin() + y_prime_low); //erase x
                }
                else if (derivat.back() == "1") //x 1 * -> x
                {
                    //                puts("hi 335");
                    derivat.pop_back(); //erase the 1
                }
                else if (derivat[y_prime_low - 1] == "1") //1 y' * -> y'
                {
                    //                puts("hi 340");
                    derivat.erase(derivat.begin() + y_prime_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x' y * x y' * */
                }
            }
            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - x_low)) && (areDerivatRangesEqual(x_prime_low, x_low, k))) //thing1 thing1 - -> 0
            {
                //            puts("hi 350");
                derivat[x_prime_low] = "0"; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest of x' y * and x y' *
            }
            else
            {
                if (derivat[x_low - 1] == "0") //0 x y' * - -> x y' * ~
                {
                    //                puts("hi 358");
                    derivat.erase(derivat.begin() + x_low - 1); //remove "0"
                    derivat.push_back("~"); //add "~" at the end
                }
                else if (derivat.back() == "0") //x' y * 0 - -> x' y *
                {
                    //                puts("hi 364");
                    derivat.pop_back(); //remove "0"
                }
                else
                {
                    derivat.push_back("-"); /* x' y * x y' * - */
                }
                for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "1") //"1 1 * /" -> ""
                {
                    //                puts("hi 377");
                    derivat.pop_back(); //remove the "1"
                }
                else
                {
                    for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y y */
                    {
                        derivat.push_back(postfix[k]);
                    }
                    derivat.push_back("*"); /* x' y * x y' * - y y * */
                    derivat.push_back("/"); /* x' y * x y' * - y y * / */
                }
            }
        }
        
        else if (postfix[up] == "^")
        {
            int k;
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y ^ (0 ln y *)' * -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
                //            puts("hi 402");
                return;
            }
            else if (derivat.back() == "1") //1 y ^ (1 ln y *)' * -> 0 (because ln(1) is 0)
            {
                derivat.back() = "0";
                //            puts("hi 407");
                return;
            }
            else
            {
                for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "0") //x 0 ^ (x ln 0 *)' * -> 0
                {
                    //                puts("hi 419");
                    derivat[x_low] = "0"; //change the first symbol of x to "0"
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
                    return;
                }
                else if (derivat.back() == "1") //x 1 ^ -> x
                {
                    //                    puts("hi 426");
                    derivat.pop_back(); //erase the 1
                }
                else
                {
                    derivat.push_back("^"); /* x y ^ */
                }
            }
            
            std::vector<std::string> postfix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> x ln y *
            postfix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                postfix_temp.push_back(postfix[k]);
            }
            postfix_temp.push_back("ln"); /* x ln  */
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x ln y */
            {
                postfix_temp.push_back(postfix[k]);
            }
            if (postfix_temp.back() == "1") //x ln 1 * -> x ln
            {
                //            puts("hi 452");
                postfix_temp.pop_back();
            }
            else
            {
                postfix_temp.push_back("*"); /* x ln y * */
            }
            setPostfixGR(postfix_temp, grasp_temp);
            derivePostfixHelper(0, postfix_temp.size() - 1, dx, postfix_temp, grasp_temp, true); /* x y ^ (x ln y *)' */
            if (derivat.back() == "0") //x y ^ 0 * -> 0
            {
                //            puts("hi 455");
                derivat[x_low] = "0"; //change the first symbol of x to "0"
                derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
            }
            else if (derivat.back() == "1") //x y ^ 1 * -> x y ^
            {
                //            puts("hi 460");
                derivat.pop_back(); //erase (x ln y *)'
            }
            else
            {
                derivat.push_back("*"); /* x y ^ (x ln y *)' * */
            }
        }
        
        else if (postfix[up] == "cos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x sin ~ * -> 0
            {
                //            puts("hi 514");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back("sin"); /* x' x sin */
            derivat.push_back("~"); /* x' x sin ~ */
            if (derivat[x_low - 1] == "1") //1 x sin ~ * -> x sin ~
            {
                //            puts("hi 526");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' x sin ~ * */
            }
        }
        
        else if (postfix[up] == "sin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x cos * -> 0
            {
                //                puts("hi 540");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back("cos"); /* x' x cos */
            if (derivat[x_low - 1] == "1") //1 x cos * -> x cos
            {
                //                puts("hi 551");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' x cos * */
            }
        }
        
        else if (postfix[up] == "sqrt")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 2 x sqrt * / -> 0
            {
                //                puts("hi 565");
                return;
            }
            derivat.push_back("2"); /* x' 2 */
            for (int k = low; k <= up-1; k++) /* x' 2 x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sqrt");    /* x' 2 x sqrt */
            derivat.push_back("*");       /* x' 2 x sqrt * */
            derivat.push_back("/");       /* x' 2 x sqrt * / */
        }
        
        else if (postfix[up] == "log" || postfix[up] == "ln")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x / -> 0
            {
                //            puts("hi 551");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]);      /* x' x */
            }
            int step = derivat.size() - x_low;
            if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) //something something / -> 1
            {
                //                puts("hi 563");
                derivat[x_prime_low] = "1"; //replace first symbol of x' with "1"
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest
                return;
            }
            
            derivat.push_back("/");               /* x' x / */
        }
        
        else if (postfix[up] == "asin" || postfix[up] == "arcsin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 1 x x * - sqrt / -> 0
            {
                //                puts("hi 610");
                return;
            }
            derivat.push_back("1"); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("*");   /* x' 1 x x * */
            derivat.push_back("-");   /* x' 1 x x * - */
            derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
            derivat.push_back("/");   /* x' 1 x x * - sqrt / */
        }
        
        else if (postfix[up] == "acos" || postfix[up] == "arccos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 1 x x * - sqrt / ~ -> 0
            {
                //            puts("hi 633");
                return;
            }
            derivat.push_back("1"); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("*");   /* x' 1 x x * */
            derivat.push_back("-");   /* x' 1 x x * - */
            derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
            derivat.push_back("/");   /* x' 1 x x * - sqrt / */
            derivat.push_back("~");   /* x' 1 x x * - sqrt / ~ */
        }
        
        else if (postfix[up] == "tanh")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == "0") //0 x sech x sech * * -> 0
            {
                //                puts("hi 657");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech"); //x' x sech
            for (int k = low; k <= up-1; k++) //x' x sech x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech"); //x' x sech x sech
            derivat.push_back("*"); //x' x sech x sech *
            if (derivat[x_low - 1] == "1") //1 x sech x sech * * -> x sech x sech * *
            {
                //                puts("hi 676");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");                 //x' x sech ~ x tanh * *
            }
        }
        
        else if (postfix[up] == "sech")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == "0") //0 x sech ~ x tanh * * -> 0
            {
                //                puts("hi 681");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech");   //x' x sech
            derivat.push_back("~");      //x' x sech ~
            for (int k = low; k <= up-1; k++) //x' x sech ~ x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("tanh");   //x' x sech ~ x tanh
            derivat.push_back("*");      //x' x sech ~ x tanh *
            if (derivat[x_low - 1] == "1") //1 x sech ~ x tanh * * -> x sech ~ x tanh *
            {
                //                puts("hi 699");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");                 //x' x sech ~ x tanh * *
            }
        }
        
        else if (postfix[up] == "exp")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x exp * -> 0
            {
                //            puts("hi 649");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]);      /* x' x */
            }
            derivat.push_back("exp");               /* x' x exp */
            if (derivat[x_low - 1] == "1") //1 x exp * -> x exp
            {
                //                puts("hi 660");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");               /* x' x exp * */
            }
        }
        
        else if (postfix[up] == "~")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "~")
            {
                derivat.pop_back(); //two unary minuses cancel each-other
            }
            else
            {
                derivat.push_back(postfix[up]); /* x' ~ */
            }
        }
        
        else
        {
            if (postfix[up] == dx)
            {
                this->derivat.push_back("1");
            }
            else
            {
                this->derivat.push_back("0");
            }
        }
    }
    
    void derivePostfix(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp)
    {
        derivePostfixHelper(low, up, dx, postfix, grasp, false);
    }
};

std::string process_str(const std::string& expression)
{
    std::string new_string;
    new_string.reserve(2*expression.size());
    for (size_t i = 0; i < expression.size(); i++)
    {
        if (expression[i] == '~')
        {
            new_string += '-';
        }
        else if (expression[i] == '^')
        {
            new_string += '*';
            new_string += '*';
        }
        else if ((i < (expression.size() - 1)) && (expression[i] == 'x') && (expression[i+1] == '0'))
        {
            new_string += 't';
            i++;
        }
        else
        {
            new_string += expression[i];
        }
    }
    return new_string;
}

struct DataRow
{
    float x0;
    float mse;
    int depth;
    std::string expression_type;
    std::string expression;
    std::string orig_expression;
    DataRow(float x0_val, float mse_val, int depth_val, std::string expression_type_val, std::string expression_val, std::string orig_expression_val)
    : x0{x0_val}, mse{mse_val}, depth{depth_val}, expression_type{expression_type_val}, expression{expression_val}, orig_expression{orig_expression_val}
    {}
    DataRow() = default;
    std::string toCSVRow() const
    {
        std::ostringstream oss;
        oss << x0 << "," << mse << "," << depth << ","
            << expression_type << "," << process_str(expression) << "," << orig_expression;
        return oss.str();
    }
};

// Global ostream operator
std::ostream& operator<<(std::ostream& os, const DataRow& row)
{
    os << "x0: " << row.x0 << ", "
       << "MSE: " << row.mse << ", "
       << "Depth: " << row.depth << ", "
       << "Expression Type: " << row.expression_type << ", "
       << "Expression: " << process_str(row.expression) << ", "
       << "Expression Orignal Format: " << row.orig_expression;
    return os;
}

void overwriteCSVRow(const DataRow& result, const DataRow& new_result)
{
    std::ifstream file_in(file_path);
    if (!file_in.is_open())
    {
        throw std::runtime_error("Could not open input file.");
    }

    std::vector<std::string> lines;
    std::string line;

    // Read the header
    if (std::getline(file_in, line))
    {
        lines.push_back(line);  // Save the header
    }

    bool found = false;

    // Read and update rows
    while (std::getline(file_in, line))
    {
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        float x0_val = std::stof(token);

        if (std::fabs(x0_val - result.x0) < 1e-6)
        {
            lines.push_back(new_result.toCSVRow());
            found = true;
        }
        else
        {
            lines.push_back(line);
        }
    }
    file_in.close();

    if (!found)
    {
        throw std::runtime_error("Matching row not found.");
    }

    // Write updated content back to file
    std::ofstream file_out(file_path, std::ios::trunc);
    if (!file_out.is_open())
    {
        throw std::runtime_error("Could not open output file.");
    }

    for (const auto& l : lines)
    {
        file_out << l << '\n';
    }
    file_out.close();
}

void addRow(const DataRow& new_result)
{
    std::ifstream infile(file_path);
    if (!infile.is_open())
    {
        throw std::runtime_error("Could not open input file.");
    }

    std::vector<std::string> lines;
    std::string line;

    // Read header
    if (std::getline(infile, line))
    {
        lines.push_back(line);  // Save header
    }

    bool foundDuplicate = false;

    // Read file and check for duplicates
    while (std::getline(infile, line))
    {
        std::istringstream ss(line);
        DataRow row(0, 0, 0, "", "", "");

        if (std::getline(ss, line, ','))
        {
            row.x0 = std::stof(line);
        }
        else
        {
            std::cerr << "Error parsing x0 in line: " << line << std::endl;
            continue;
        }

        if (std::getline(ss, line, ','))
        {
            row.mse = std::stof(line);
        }
        else
        {
            std::cerr << "Error parsing mse in line: " << line << std::endl;
            continue;
        }

        if (std::getline(ss, line, ','))
        {
            row.depth = std::stoi(line);
        }
        else
        {
            std::cerr << "Error parsing depth in line: " << line << std::endl;
            continue;
        }

        if (!std::getline(ss, row.expression_type, ',') ||
            !std::getline(ss, row.expression, ',') ||
            !std::getline(ss, row.orig_expression))
        {
            std::cerr << "Error parsing remaining fields in line: " << line << std::endl;
            continue;
        }


        if (std::fabs(row.x0 - new_result.x0) < 1e-6)
        {
            if (new_result.mse < row.mse)
            {
                lines.push_back(new_result.toCSVRow());
            }
            else
            {
                lines.push_back(row.toCSVRow());
            }
            foundDuplicate = true;
        }
        else
        {
            lines.push_back(row.toCSVRow());
        }
    }
    infile.close();

    // Append if no duplicate found
    if (!foundDuplicate)
    {
        lines.push_back(new_result.toCSVRow());
    }

    // Write updated content back to the file
    std::ofstream outfile(file_path, std::ios::trunc);
    if (!outfile.is_open())
    {
        throw std::runtime_error("Could not open output file.");
    }

    for (const auto& l : lines)
    {
        outfile << l << '\n';
    }
    outfile.close();
}

//https://dl.acm.org/doi/pdf/10.1145/3449639.3459345?casa_token=Np-_TMqxeJEAAAAA:8u-d6UyINV6Ex02kG9LthsQHAXMh2oxx3M4FG8ioP0hGgstIW45X8b709XOuaif5D_DVOm_FwFo
//https://core.ac.uk/download/pdf/6651886.pdf
DataRow SimulatedAnnealing(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false, const std::vector<std::string>& seed_expression = {})
{
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    
    /*
     Outside of thread:
     */
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression, best_second_deriv_expression, orig_second_deriv_expression;

    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_second_deriv_expression, &orig_second_deriv_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE, &seed_expression]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token, false);
        
        sync_point.arrive_and_wait();
        Board secondary(false, 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token, false); //For perturbations
        float score = 0.0f, check_point_score = 0.0f;
        
        std::vector<std::string> current;
        std::vector<std::pair<int, int>> sub_exprs;
        std::vector<std::string> temp_legal_moves;
        std::uniform_int_distribution<int> rand_depth_dist(0, x.n);
        size_t temp_sz;
//        std::string expression, orig_expression, best_expression;
        constexpr float T_max = 0.1f;
        constexpr float T_min = 0.012f;
        constexpr float ratio = T_min/T_max;
        float T = T_max;
        
        auto P = [&](float delta)
        {
            return exp(delta/T);
        };
        
        auto updateScore = [&](float r = 1.0f)
        {
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
            if ((score > max_score) || (x.pos_dist(generator) < P(score-max_score)))
            {
                current = x.pieces; //update current expression
                if (score > max_score)
                {
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_MSE = x.MSE_curr;
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                    best_second_deriv_expression = x._to_infix(x.derivat2);
                    orig_second_deriv_expression = x.expression(x.derivat2);
                    
                    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
                    std::cout << "Best expression = " << best_expression << '\n';
                    std::cout << "Best expression (original format) = " << orig_expression << '\n';
                    std::cout << "Best second derivative expression = " << best_second_deriv_expression << '\n';
                    std::cout << "Best second derivative expression (original format) = " << orig_second_deriv_expression << '\n';
                }
            }
            else
            {
                x.pieces = current; //reset perturbed state to current state
            }
            T = r*T;
        };
        
        //Another way to do this might be clustering...
        auto Perturbation = [&](int n, int i)
        {
            //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
            secondary.pieces.clear();
            sub_exprs.clear();
            secondary.n = n;
            while (secondary.complete_status() == -1)
            {
                temp_legal_moves = secondary.get_legal_moves();
                std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                secondary.pieces.push_back(temp_legal_moves[distribution(generator)]);
            }
            
//            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).first == secondary.n);
//            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).second);
            
            if (n == x.n)
            {
                std::swap(secondary.pieces, x.pieces);
            }
            else
            {
                //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                //in `x.pieces` and store them in an std::vector<std::pair<int, int>>
                //called `sub_exprs`.
                secondary.get_indices(sub_exprs, x.pieces);
                
                //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `pert_ind`

                std::uniform_int_distribution<int> distribution(0, sub_exprs.size() - 1);
                int pert_ind = distribution(generator);
                
                //Step 4: Substitute sub_exprs_1[pert_ind] in x.pieces with secondary_one.pieces
                
                auto start = x.pieces.begin() + sub_exprs[pert_ind].first;
                auto end = std::min(x.pieces.begin() + sub_exprs[pert_ind].second, x.pieces.end());
                x.pieces.erase(start, end+1);
                x.pieces.insert(start, secondary.pieces.begin(), secondary.pieces.end()); //could be a move operation: secondary.pieces doesn't need to be in a defined state after this->params
            }
            
            //Step 5: Reset const token labels in pieces
            size_t const_counter = 0;
            for (std::string& token: x.pieces)
            {
                if (token.substr(0, 5) == "const")
                {
                    token = "const" + std::to_string(const_counter++);
                }
            }
            
            //Step 6: Evaluate the new mutated `x.pieces` and update score if needed
            score = x.complete_status(false);
            updateScore(pow(ratio, 1.0f/(i+1)));
        };

        //Step 1: generate a random expression
        if (seed_expression.empty())
        {
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                current.push_back(x.pieces.back());
            }
        }
        else
        {
            assert(((x.expression_type == "prefix") ? x.getPNdepth(seed_expression) : x.getRPNdepth(seed_expression)).first == x.n);
            assert(((x.expression_type == "prefix") ? x.getPNdepth(seed_expression) : x.getRPNdepth(seed_expression)).second);
            x.pieces = seed_expression;
            current = x.pieces;
            score = x.complete_status(false);
            assert(score > 0);
        }
        updateScore();
        
        for (int i = 0; ((timeElapsedSince(start_time) < time) && (!stop_flag.load())); i++)
        {
            if (i && (i%50000 == 0))
            {
//                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                if (check_point_score == max_score)
                {
                    T = std::min(T*10.0f, T_max);
                }
                else
                {
                    T = std::max(T/10.0f, T_min);
                }
                check_point_score = max_score;
            }
            Perturbation(rand_depth_dist(generator), i);
            
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best second derivative expression = " << best_second_deriv_expression << '\n';
    std::cout << "Best second derivative expression (original format) = " << orig_second_deriv_expression << '\n';
    
    return result;
}
//
////https://arxiv.org/abs/2310.06609
DataRow GP(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    
    /*
     Outside of thread:
     */
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token);
        sync_point.arrive_and_wait();
        Board secondary_one(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token), secondary_two(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token); //For crossover and mutations
        float score = 0.0f, mut_prob = 0.8f, rand_mut_cross;
        constexpr int init_population = 2000;
        std::vector<std::pair<std::vector<std::string>, float>> individuals;
        std::pair<std::vector<std::string>, float> individual_1, individual_2;
        std::vector<std::pair<int, int>> sub_exprs_1, sub_exprs_2;
        individuals.reserve(2*init_population);
        std::vector<std::string> temp_legal_moves;
        std::uniform_int_distribution<int> rand_depth_dist(0, x.n - 1), selector_dist(0, init_population - 1);
        int rand_depth, rand_individual_idx_1, rand_individual_idx_2;
        std::uniform_real_distribution<float> rand_mut_cross_dist(0.0f, 1.0f);
        size_t temp_sz;
    //    std::string expression, orig_expression, best_expression;
        
        auto updateScore = [&]()
        {
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
                std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
                
//                std::cout << "Derivative = " << x._to_infix(x.derivat) << '\n';
            }
        };
        
        //Step 1, generate init_population expressions
        for (int i = 0; i < init_population; i++)
        {
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                assert(temp_sz);
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
            }
            updateScore();
            individuals.push_back(std::make_pair(x.pieces, score));
            x.pieces.clear();
        }
        
        auto Mutation = [&](int n)
        {
            //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
            secondary_one.pieces.clear();
            sub_exprs_1.clear();
            secondary_one.n = n;
            while (secondary_one.complete_status() == -1)
            {
                temp_legal_moves = secondary_one.get_legal_moves();
                std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                secondary_one.pieces.push_back(temp_legal_moves[distribution(generator)]);
            }
            
            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).first == secondary_one.n);
            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).second);

            //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
            //in `x.pieces` and store them in an std::vector<std::pair<int, int>>
            //called `sub_exprs_1`.
            x.pieces = individuals[selector_dist(generator)].first; //A randomly selected individual to be mutated
            secondary_one.get_indices(sub_exprs_1, x.pieces);
            
            //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `mut_ind`
            std::uniform_int_distribution<int> distribution(0, sub_exprs_1.size() - 1);
            int mut_ind = distribution(generator);
            
            //Step 4: Substitute sub_exprs_1[mut_ind] in x.pieces with secondary_one.pieces
            
            auto start = x.pieces.begin() + sub_exprs_1[mut_ind].first;
            auto end = std::min(x.pieces.begin() + sub_exprs_1[mut_ind].second, x.pieces.end()-1);
            x.pieces.erase(start, end+1);
            x.pieces.insert(start, secondary_one.pieces.begin(), secondary_one.pieces.end());
            
            //Step 5: Reset const token labels in pieces
            size_t const_counter = 0;
            for (std::string& token: x.pieces)
            {
                if (token.substr(0, 5) == "const")
                {
                    token = "const" + std::to_string(const_counter++);
                }
            }
            
            //Step 6: Evaluate the new mutated `x.pieces` and update score if needed
            score = x.complete_status(false);
            updateScore();
            individuals.push_back(std::make_pair(x.pieces, score));
        };
        
        auto Crossover = [&](int n)
        {
            sub_exprs_1.clear();
            sub_exprs_2.clear();
            secondary_one.n = n;
            secondary_two.n = n;
            
            rand_individual_idx_1 = selector_dist(generator);
            assert(individuals.size() && rand_individual_idx_1 < individuals.size());
            individual_1 = individuals[rand_individual_idx_1];
            
            do {
                rand_individual_idx_2 = selector_dist(generator);
            } while (rand_individual_idx_2 == rand_individual_idx_1);
            assert(individuals.size() && rand_individual_idx_1 < individuals.size());
            individual_2 = individuals[rand_individual_idx_2];
        
            //Step 1: Identify the starting and stopping index pairs of all depth-n sub-expressions
            //in `individual_1.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_1`.
            secondary_one.get_indices(sub_exprs_1, individual_1.first);
            
            //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
            //in `individual_2.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_2`.
            secondary_two.get_indices(sub_exprs_2, individual_2.first);
            
            //Step 3: Generate a random uniform int from 0 to sub_exprs_1.size() - 1 called `mut_ind_1`
            std::uniform_int_distribution<int> distribution_1(0, sub_exprs_1.size() - 1);
            int mut_ind_1 = distribution_1(generator);
            
            //Step 4: Generate a random uniform int from 0 to sub_exprs_2.size() - 1 called `mut_ind_2`
            std::uniform_int_distribution<int> distribution_2(0, sub_exprs_2.size() - 1);
            int mut_ind_2 = distribution_2(generator);
            
            //Step 5: Swap sub_exprs_1[mut_ind_1] in individual_1.first with sub_exprs_2[mut_ind_2] in individual_2.first
            auto start_1 = individual_1.first.begin() + sub_exprs_1[mut_ind_1].first;
            auto end_1 = std::min(individual_1.first.begin() + sub_exprs_1[mut_ind_1].second, individual_1.first.end());
            
            auto start_2 = individual_2.first.begin() + sub_exprs_2[mut_ind_2].first;
            auto end_2 = std::min(individual_2.first.begin() + sub_exprs_2[mut_ind_2].second, individual_2.first.end());
            
    //        insert the range start_2, end_2+1 into individual_1 and the range start_1, end_1+1 into individual_2.
            
            if ((end_1 - start_1) < (end_2 - start_2))
            {
                std::swap_ranges(start_1, end_1+1, start_2);
                //Insert remaining part of sub_individual_2.first into individual_1.first
                individual_1.first.insert(end_1+1, start_2 + (end_1+1-start_1), end_2+1);
                //Remove the remaining part of sub_individual_2.first from individual_2.first
                individual_2.first.erase(start_2 + (end_1+1-start_1), end_2+1);
            }
            else if ((end_2 - start_2) < (end_1 - start_1))
            {
                std::swap_ranges(start_2, end_2+1, start_1);
                //Insert remaining part of sub_individual_1.first into individual_2.first
                individual_2.first.insert(end_2+1, start_1 + (end_2+1-start_2), end_1+1);
                //Remove the remaining part of sub_individual_1.first from individual_1.first
                individual_1.first.erase(start_1 + (end_2+1-start_2), end_1+1);
            }
            else
            {
                std::swap_ranges(start_1, end_1+1, start_2);
            }
            
            //Step 6: Reset const token labels in individual_1.first
            size_t const_counter = 0;
            for (std::string& token: individual_1.first)
            {
                if (token.substr(0, 5) == "const")
                {
                    token = "const" + std::to_string(const_counter++);
                }
            }
            
            //Step 7: Evaluate the new `x.pieces` and update score if needed
            x.pieces = individual_1.first;
            score = x.complete_status(false);
            updateScore();
            
            individuals.push_back(std::make_pair(x.pieces, score));
            
            //Step 8: Reset const token labels in individual_2.first
            const_counter = 0;
            for (std::string& token: individual_2.first)
            {
                if (token.substr(0, 5) == "const")
                {
                    token = "const" + std::to_string(const_counter++);
                }
            }
            
            //Step 9: Evaluate the new `x.pieces` and update score if needed
            x.pieces = individual_2.first;
            score = x.complete_status(false);
            updateScore();
            
            individuals.push_back(std::make_pair(x.pieces, score));
        };

        
        for (/*int ngen = 0*/; ((timeElapsedSince(start_time) < time) && (!stop_flag.load())); /*ngen++*/)
        {
//            if (ngen && (ngen%5 == 0))
//            {
//                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
//            }
            //Produce N additional individuals through crossover and mutation
            for (int n = 0; n < init_population; n++)
            {
                //Step 1: Generate a random number between 0 and 1 called `rand_mut_cross`
                rand_mut_cross = rand_mut_cross_dist(generator);
                
                //Step 2: Generate a random uniform int from 0 to x.n - 1 called `rand_depth`
                rand_depth = rand_depth_dist(generator);
                
                //Step 4: Call Mutation function if 0 <= rand_mut_cross <= mut_prob, else select Crossover
                if (rand_mut_cross <= mut_prob)
                {
                    Mutation(rand_depth);
                }
                else
                {
                    Crossover(rand_depth);
                }
            }
            std::sort(individuals.begin(), individuals.end(),
            [](std::pair<std::vector<std::string>, float>& individual_1, std::pair<std::vector<std::string>, float>& individual_2)
            {
                return individual_1.second > individual_2.second;
            });
            individuals.resize(init_population);
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
    
    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    
    return result;
}

DataRow PSO(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    
    /*
     Outside of thread:
     */
    
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression;
    
    auto start_time = Clock::now();

    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token);
        
        sync_point.arrive_and_wait();
        float score = 0, check_point_score = 0;
        std::vector<std::string> temp_legal_moves;
        
        size_t temp_sz;
    //    std::string expression, orig_expression, best_expression;
        
        /*
         For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
         */
        std::vector<float> particle_positions, best_positions, v, curr_positions;
        particle_positions.reserve(x.reserve_amount); //stores record of all current particle position indices
        best_positions.reserve(x.reserve_amount); //indices corresponding to best pieces
        curr_positions.reserve(x.reserve_amount); //indices corresponding to x.pieces
        v.reserve(x.reserve_amount); //stores record of all current particle velocities
        float rp, rg, new_v, c = 0.0f;
        int c_count = 0;
        std::unordered_map<float, std::unordered_map<int, int>> Nsa;
        std::unordered_map<float, std::unordered_map<int, float>> Psa;
        std::unordered_map<int, float> p_i_vals, p_i;
        
        /*
         In this implementation of PSO:
         
             The traditional PSO initializes the particle positions to be between 0 and 1. However, in this application,
             the particle positions are discrete values and any of the legal integer tokens (moves). The
             velocities are continuous-valued and perturb the postions, which are subsequently constrained by rounding to
             the nearest whole number then taking the modulo w.r.t. the # of allowed legal moves.
         
         */
        
        for (int iter = 0; ((timeElapsedSince(start_time) < time) && (!stop_flag.load())); iter++)
        {
            if (iter && (iter%50000 == 0))
            {
    //            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
    //            std::cout << "check_point_score = " << check_point_score
    //            << ", max_score = " << max_score << ", c = " << c << '\n';
                if (check_point_score == max_score)
                {
                    c_count++;
                    std::uniform_real_distribution<float> temp(-c_count, c_count);
    //                std::cout << "c: " << c << " -> ";
                    c = temp(generator);
    //                std::cout << c << '\n';
                }
                else
                {
    //                std::cout << "c: " << c << " -> ";
                    c = 0.0f; //if new best found, reset c and try to exploit the new best
                    c_count = 0;
    //                std::cout << c << '\n';
                }
                check_point_score = max_score;
            }
            
            for (int i = 0; (score = x.complete_status()) == -1; i++) //i is the index of the token
            {
                rp = x.pos_dist(generator), rg = x.pos_dist(generator);
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves

                if (i == static_cast<int>(particle_positions.size())) //Then we need to create a new particle with some initial position and velocity
                {
                    particle_positions.push_back(x.pos_dist(generator));
                    v.push_back(x.vel_dist(generator));
                }
                
                particle_positions[i] = trueMod(std::round(particle_positions[i]), temp_sz);
                x.pieces.push_back(temp_legal_moves[particle_positions[i]]); //x.pieces holds the pieces corresponding to the indices
                curr_positions.push_back(particle_positions[i]);
                if (i == static_cast<int>(best_positions.size()))
                {
                    best_positions.push_back(x.pos_dist(generator));
                    best_positions[i] = trueMod(std::round(best_positions[i]), temp_sz);
                }
                //https://hal.science/hal-00764996
                //https://www.researchgate.net/publication/216300408_An_off-the-shelf_PSO
                new_v = (0.721*v[i] + x.phi_1*rg*(best_positions[i] - particle_positions[i]) + x.phi_2*rp*(p_i[i] - particle_positions[i]) + c);
                v[i] = copysign(std::min(new_v, FLT_MAX), new_v);
                particle_positions[i] += v[i];
                Nsa[curr_positions[i]][i]++;
            }
            
            for (int i = 0; i < static_cast<int>(curr_positions.size()); i++)
            {
                Psa[curr_positions[i]][i] = (Psa[curr_positions[i]][i]+score)/Nsa[curr_positions[i]][i];
                if (Psa[curr_positions[i]][i] > p_i_vals[i])
                {
                    p_i[i] = curr_positions[i];
                }
                p_i_vals[i] = std::max(p_i_vals[i], Psa[curr_positions[i]][i]);
                
            }
            
            if (score > max_score)
            {
                for (int idx = 0; idx < static_cast<int>(curr_positions.size()); idx++)
                {
                    best_positions[idx] = curr_positions[idx];
                }
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
            }
            x.pieces.clear();
            curr_positions.clear();
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
        
    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    
    return result;
}

//https://arxiv.org/abs/2205.13134
DataRow ConcurrentMCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    
    /*
     Outside of thread:
     */
    std::atomic<float> max_score{0.0f};
    std::atomic<float> best_MSE{FLT_MAX};
    
    std::string best_expression, orig_expression;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, float>> Qsa;
    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, int>> Nsa;
    boost::concurrent_flat_map<std::string, int> Ns;
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE, &Qsa, &Nsa, &Ns]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev());
        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token);
        
        sync_point.arrive_and_wait();
        float score = 0.0f, check_point_score = 0.0f, UCT, UCT_best;
        std::string best_act;
        
        std::vector<std::string> temp_legal_moves;
        std::string state;
        
        float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
        std::vector<std::pair<std::string, std::string>> moveTracker;
        moveTracker.reserve(x.reserve_amount);
        temp_legal_moves.reserve(x.reserve_amount);
        state.reserve(2*x.reserve_amount);
        //        double str_convert_time = 0.0;
        auto getString  = [&]()
        {
            if (!x.pieces.empty())
            {
                state += x.pieces[x.pieces.size()-1] + " ";
            }
        };
        
        for (int i = 0; ((timeElapsedSince(start_time) < time) && (!stop_flag.load())); i++)
        {
            if (i && (i%1000 == 0))
            {
                //                    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                //                    std::cout << "check_point_score = " << check_point_score
                //                    << ", max_score = " << max_score << ", c = " << c << '\n';
                if (check_point_score == max_score)
                {
                    //                        std::cout << "c: " << c << " -> ";
                    c += 1.4;
                    //                        std::cout << c << '\n';
                }
                else
                {
                    //                        std::cout << "c: " << c << " -> ";
                    c = 1.4; //if new best found, reset c and try to exploit the new best
                    //                        std::cout << c << '\n';
                    check_point_score = max_score;
                }
            }
            state.clear();
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves();
                assert(temp_legal_moves.size());
                
//                for (float i: temp_legal_moves)
//                {
//                    assert(i >= 0.0f);
//                }
//                    auto start_time = Clock::now();
                getString();
//                    str_convert_time += timeElapsedSince(start_time);
                UCT = 0.0f;
                UCT_best = -FLT_MAX;
                best_act = temp_legal_moves[0];
                std::vector<std::string> best_acts;
                best_acts.reserve(temp_legal_moves.size());
                
                for (const std::string& a : temp_legal_moves)
                {
//                    assert(a > -1.0f);
//                    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<float, float>>
                    if (Nsa.contains(state))
                    {
                        int Nsa_contains_a = 0;
                        Nsa.cvisit(state, [&](const auto& x)
                        {
                            if (x.second.contains(a))
                            {
                               x.second.cvisit(a, [&](const auto& y)
                               {
                                   Nsa_contains_a = y.second;
                               });
                            }
                        });
                        if (Nsa_contains_a)
                        {
                            float Qsa_s_a;
                            int Ns_s, Nsa_s_a;
                            Qsa.cvisit(state, [&](const auto& x)
                            {
                                x.second.cvisit(a, [&](const auto& y)
                                {
                                    Qsa_s_a = y.second;
                                });
                            });
                            Nsa.cvisit(state, [&](const auto& x)
                            {
                                x.second.cvisit(a, [&](const auto& y)
                                {
                                    Nsa_s_a = y.second;
                                });
                            });
                            Ns.cvisit(state, [&](const auto& x)
                            {
                                Ns_s = x.second;
                            });
                            UCT = Qsa_s_a + c*sqrt(log(Ns_s)/Nsa_s_a);
                        }
                        else
                        {
                            Nsa.visit(state, [&](auto& x)
                            {
                                x.second.insert_or_assign(a, 0);
                            });
                            Qsa.visit(state, [&](auto& x)
                            {
                               x.second.insert_or_assign(a, 0.0f);
                            });
                            Ns.insert_or_assign(state, 0);
                            best_acts.push_back(a);
                            UCT = -FLT_MAX;
                        }
                    }
                    else
                    {
                        Nsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, int>({{a, 0}}));
                        Qsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, float>({{a, 0.0f}}));
                        Ns.insert_or_assign(state, 0);
                        best_acts.push_back(a);
                        UCT = -FLT_MAX;
                    }
                    
                    if (UCT > UCT_best)
                    {
                        best_act = a;
                        UCT_best = UCT;
                    }
                }
//                assert(best_acts.size() || (best_act > -1.0f));
                if (best_acts.size())
                {
                    std::uniform_int_distribution<int> distribution(0, best_acts.size() - 1);
                    best_act = best_acts[distribution(generator)];
                }
                
                x.pieces.push_back(best_act);
                moveTracker.push_back(make_pair(state, best_act));
//                assert(Ns.contains(state));
                Ns.visit(state, [&](auto& x)
                {
                    x.second++;
                });
//                assert(Nsa.contains(state));
                Nsa.visit(state, [&](auto& x)
                {
                    if (!x.second.contains(best_act))
                    {
                        x.second.insert_or_assign(best_act, 0);
                    }
//                    assert( x.second.contains(best_act));
                    x.second.visit(best_act, [&](auto& y)
                    {
                       y.second++;
                    });
                });
            }
            //backprop reward `score`
            for (auto& state_action: moveTracker)
            {
//                assert(Qsa.contains(state_action.first));
                Qsa.visit(state_action.first, [&](auto& x)
                {
//                    assert(x.second.contains(state_action.second));
                    if (!x.second.contains(state_action.second))
                    {
                        Nsa.visit(state, [&](auto& y)
                        {
                            y.second.insert_or_assign(state_action.second, 0);
                        });
                        x.second.insert_or_assign(state_action.second, 0.0f);
                    }
                    
                    x.second.visit(state_action.second, [&](auto& y)
                    {
                        y.second = std::max(y.second, score);
                    });
                });
            }
            
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
            }
            x.pieces.clear();
            moveTracker.clear();
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
    
    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    
    return result;
}

//https://arxiv.org/abs/2205.13134
DataRow MCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    
    /*
     Outside of thread:
     */
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev());
        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token);
        
        sync_point.arrive_and_wait();
        float score = 0.0f, check_point_score = 0.0f, UCT, UCT_best;
        std::string best_act;
        
        std::vector<std::string> temp_legal_moves;
        std::unordered_map<std::string, std::unordered_map<std::string, float>> Qsa, Nsa;
        std::unordered_map<std::string, float> Ns;
        std::string state;
        
        float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
        std::vector<std::pair<std::string, std::string>> moveTracker;
        moveTracker.reserve(x.reserve_amount);
        temp_legal_moves.reserve(x.reserve_amount);
        state.reserve(2*x.reserve_amount);
        //        double str_convert_time = 0.0;
        auto getString  = [&]()
        {
            if (!x.pieces.empty())
            {
                state += (x.pieces[x.pieces.size()-1] + " ");
            }
        };
        
        for (int i = 0; ((timeElapsedSince(start_time) < time) && (!stop_flag.load())); i++)
        {
            if (i && (i%500 == 0))
            {
                //                    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                //                    std::cout << "check_point_score = " << check_point_score
                //                    << ", max_score = " << max_score << ", c = " << c << '\n';
                if (check_point_score == max_score)
                {
                    //                        std::cout << "c: " << c << " -> ";
                    c += 1.4;
                    //                        std::cout << c << '\n';
                }
                else
                {
                    //                        std::cout << "c: " << c << " -> ";
                    c = 1.4; //if new best found, reset c and try to exploit the new best
                    //                        std::cout << c << '\n';
                    check_point_score = max_score;
                }
            }
            state.clear();
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves();
                assert(temp_legal_moves.size());
//                    auto start_time = Clock::now();
                getString();
//                    str_convert_time += timeElapsedSince(start_time);
                UCT = 0.0f;
                UCT_best = -FLT_MAX;
                best_act = temp_legal_moves[0];
                std::vector<std::string> best_acts;
                best_acts.reserve(temp_legal_moves.size());
                
                for (const std::string& a : temp_legal_moves)
                {
                    if (Nsa[state].count(a))
                    {
                        UCT = Qsa[state][a] + c*sqrt(log(Ns[state])/Nsa[state][a]);
                    }
                    else
                    {
                        //not explored -> explore it.
                        best_acts.push_back(a);
                        UCT = -FLT_MAX;
                    }
                    
                    if (UCT > UCT_best)
                    {
                        best_act = a;
                        UCT_best = UCT;
                    }
                }
                
                if (best_acts.size())
                {
                    std::uniform_int_distribution<int> distribution(0, best_acts.size() - 1);
                    best_act = best_acts[distribution(generator)];
                }
                x.pieces.push_back(best_act);
                moveTracker.push_back(make_pair(state, best_act));
                Ns[state]++;
                Nsa[state][best_act]++;
            }
            //backprop reward `score`
            for (auto& state_action: moveTracker)
            {
                Qsa[state_action.first][state_action.second] = std::max(Qsa[state_action.first][state_action.second], score);
            }
            
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
                std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
            }
            x.pieces.clear();
            moveTracker.clear();
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
    
    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    
    return result;
}

DataRow RandomSearch(const Eigen::MatrixXf& data, const int depth = 3, const std::string expression_type = "prefix", const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, float upperVar = 0.02, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    std::signal(SIGINT, signalHandler); // Register signal handler
    std::signal(SIGINT, signalHandler); // Register signal handler

    /*
     Outside of thread:
     */
    
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression, best_second_deriv_expression, orig_second_deriv_expression;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_second_deriv_expression, &orig_second_deriv_expression, &const_tokens, &isConstTol, &upperVar, &const_token, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator

        Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, upperVar, const_token);
//        {
//            x.pieces = {"x0", "sech", "x0", "sech", "~", "x0", "tanh", "*", "*", "x0", "sech", "~", "x0", "tanh", "*", "x0", "sech", "*", "+", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "*", "x0", "sech", "x0", "sech", "*", "x0", "tanh", "x0", "sech", "x0", "sech", "*", "*", "x0", "sech", "x0", "sech", "*", "x0", "tanh", "*", "+", "~", "2", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "*", "/", "*", "-", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "*", "/", "2", "x0", "tanh", "asin", "sqrt", "*", "*", "x0", "sech", "x0", "sech", "*", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "/", "2", "x0", "sech", "x0", "sech", "*", "1", "x0", "tanh", "x0", "tanh", "*", "-", "sqrt", "/", "2", "x0", "tanh", "asin", "sqrt", "*", "/", "*", "*", "-", "2", "x0", "tanh", "asin", "sqrt", "*", "2", "x0", "tanh", "asin", "sqrt", "*", "*", "/"};
//            Eigen::VectorXf params;
//            std::cout << x.expression_evaluator(x.params, x.pieces) << '\n';
//            std::cout << std::boolalpha << x.expression_evaluator_valid_flag << '\n';
//            exit(1);
//        }
        
        sync_point.arrive_and_wait();
        float score = 0.0f;
        std::vector<std::string> temp_legal_moves;
        size_t temp_sz;
        while ((timeElapsedSince(start_time) < time) && (!stop_flag.load()))
        {
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                
                assert(temp_sz);
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                {
                    x.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                }
            }
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);

            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_second_deriv_expression = x._to_infix(x.derivat2);
                orig_second_deriv_expression = x.expression(x.derivat2);
                
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
                std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
                std::cout << "Best second derivative expression = " << best_second_deriv_expression << '\n';
                std::cout << "Best second derivative expression (original format) = " << orig_second_deriv_expression << '\n';
                
            }
            x.pieces.clear();
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
    
    DataRow result(InvertedPendulum::x_0, best_MSE, depth, expression_type, best_expression, orig_expression);
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best second derivative expression = " << best_second_deriv_expression << '\n';
    std::cout << "Best second derivative expression (original format) = " << orig_second_deriv_expression << '\n';
    
    return result;
}

DataRow findClosestRow(float target)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file.");
    }

    std::string line;
    std::getline(file, line);  // Skip header

    DataRow closestRow;
    float minDiff = std::numeric_limits<float>::max();

    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        DataRow row;
        std::string depthStr;

        if (std::getline(ss, line, ',')) row.x0 = std::stof(line);
        if (std::getline(ss, line, ',')) row.mse = std::stof(line);
        if (std::getline(ss, line, ',')) row.depth = std::stoi(line);
        if (std::getline(ss, line, ',')) row.expression_type = line;
        if (std::getline(ss, line, ',')) row.expression = line;
        if (std::getline(ss, row.orig_expression, ','))
        {
            float diff = std::fabs(row.x0 - target);
            if (diff < minDiff)
            {
                minDiff = diff;
                closestRow = row;
            }
        }
    }

    file.close();
    return closestRow;
}

int main(int argc, char* argv[])
{
    DataRow new_result;
    constexpr double time = 10000;

//    new_result = RandomSearch(createLinspaceMatrix(1000, 1, {0.0f}, {InvertedPendulum::T}) /*data used to solve differential equation*/, 3 /*fixed depth of generated solutions*/, "postfix" /*expression representation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "autodiff" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, 1.0e-5f /*lower limit for variance of solution*/, 0.01 /*upper limit for variance of solution*/, true /*whether to include "const" token to be optimized, though `const_tokens` must be true as well*/);
//    new_result = SimulatedAnnealing(createLinspaceMatrix(1000, 1, {0.0f}, {InvertedPendulum::T}) /*data used to solve differential equation*/, 3 /*fixed depth of generated solutions*/, "postfix" /*expression representation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "autodiff" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, 1.0e-5f /*lower limit for variance of solution*/, 0.01 /*upper limit for variance of solution*/, true /*whether to include "const" token to be optimized, though `const_tokens` must be true as well*/, split("x0 tanh 100.000000 log * cos") /*seed expression for simulated annealing*/);
    new_result = SimulatedAnnealing(createLinspaceMatrix(1000, 1, {0.0f}, {InvertedPendulum::T}) /*data used to solve differential equation*/, 3 /*fixed depth of generated solutions*/, "prefix" /*expression representation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "autodiff" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, 1.0e-5f /*lower limit for variance of solution*/, 0.01 /*upper limit for variance of solution*/, true /*whether to include "const" token to be optimized, though `const_tokens` must be true as well*/, split("cos * tanh x0 log 100.000000") /*seed expression for simulated annealing*/);
    
    return 0;
}
//cos * tanh x0 log 100.000000
//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator

//g++ -Wall -std=c++20 -o HorizontalControlInvertedPendulum HorizontalControlInvertedPendulum.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//g++ -Wall -std=c++20 -o HorizontalControlInvertedPendulum HorizontalControlInvertedPendulum.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
