#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>
#include <cmath>
#include <algorithm>


using namespace std;

struct distance_single_sample {
    double distance;
    int label;
};

void quick_sort(vector<distance_single_sample> &vec, int low, int high);
// Reads a flattened (NUM_SAMPLES x FEATURE_DIM) binary file of floats
vector<vector<float>> read_features(const string &filename, size_t num_samples, size_t feature_dim) {
    size_t total_elements = num_samples * feature_dim;
    vector<float> flat(total_elements);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(flat.data()), total_elements * sizeof(float));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    // Reshape the flat vector into a vector of vectors.
    vector<vector<float>> features;
    features.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        vector<float> sample(flat.begin() + i * feature_dim, flat.begin() + (i + 1) * feature_dim);
        features.push_back(move(sample));
    }
    
    return features;
}

// Reads a binary file containing labels.
vector<int> read_labels(const string &filename, size_t num_samples) {
    vector<int> labels(num_samples);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(labels.data()), num_samples * sizeof(int));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    return labels;
}

int find_majority_label(const vector<distance_single_sample>& sorted_distances, int k) {
    const int num_classes = 10; 
    vector<int> votes(num_classes, 0);
    
    int num_neighbors_to_check = min(k, static_cast<int>(sorted_distances.size()));

    for (int i = 0; i < num_neighbors_to_check; ++i) {
        int label = sorted_distances[i].label;
        if (label >= 0 && label < num_classes) {
            votes[label]++;
        }
    }

    int predicted_label = 0;
    int max_votes = -1;

    for (int label_index = 0; label_index < num_classes; ++label_index) {
        if (votes[label_index] > max_votes) {
            max_votes = votes[label_index];
            predicted_label = label_index;
        }
    }

    return predicted_label;
}

int partition(vector<distance_single_sample> &vec, int low, int high) {
    if (low >= high || low < 0 || high >= static_cast<int>(vec.size())) {
        return low;
    }

    double pivot_distance = vec[high].distance;
    int i = (low - 1);

    for (int j = low; j < high; j++) {
        if (vec[j].distance < pivot_distance) {
            i++;
            swap(vec[i], vec[j]);
        }
    }

    swap(vec[i + 1], vec[high]);
    return (i + 1);
}

void quick_sort_parallel(vector<distance_single_sample> &vec, int low, int high, int depth = 0) {
    const int MAX_TASK_DEPTH = 3;  // Control task creation depth to avoid excessive overhead
    
    if (low < high) {
        // Find partition index
        int pi = partition(vec, low, high);
        
        if (depth >= MAX_TASK_DEPTH) {
            // Sequential execution for deeper recursion levels
            quick_sort(vec, low, pi - 1);
            quick_sort(vec, pi + 1, high);
        } else {
            // Parallel execution using tasks for shallow recursion
            #pragma omp task
            {
                quick_sort_parallel(vec, low, pi - 1, depth + 1);
            }
            
            #pragma omp task
            {
                quick_sort_parallel(vec, pi + 1, high, depth + 1);
            }
            
            #pragma omp taskwait
        }
    }
}

vector<int> run_knn_parallel(vector<vector<float>>& features_train, vector<int>& labels_train, 
    vector<vector<float>>& features_test, 
    size_t NUM_SAMPLES_TRAIN, size_t FEATURE_DIM, 
    size_t NUM_TEST_SAMPLES, int k, int num_threads) {
    vector<int> all_predictions(NUM_TEST_SAMPLES);  // Allocate space for all predictions

    cout << "Processing Parallel KNN for K=" << k << " with " << num_threads << " threads..." << endl;

    // Set number of threads
    omp_set_num_threads(num_threads);

    // Timing variables to measure performance
    double total_dist_time_acc = 0.0;
    double total_sort_time_acc = 0.0;
    double start_overall_time = omp_get_wtime();

    // Setup progress tracking - use atomic for thread-safe updates
    int total_processed = 0;

    // Process test samples in parallel
    #pragma omp parallel
    {
        vector<pair<double, double>> thread_timings;  // Store timing for each thread
        int thread_progress = 0;  // Local progress counter for each thread

        // Process chunks of test samples in parallel
        #pragma omp for schedule(dynamic, 10) nowait
        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
            // Initialize distances for current test sample
            vector<distance_single_sample> curr_dist;
            curr_dist.reserve(NUM_SAMPLES_TRAIN);  // Pre-allocate for efficiency

            // Calculate distances to all training samples
            double iter_dist_start = omp_get_wtime();

            // Calculate distances to all training samples
            for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j) {
                double sum = 0.0;

                // Compute Euclidean distance - this inner loop is a good candidate for SIMD
                #pragma omp simd reduction(+:sum)
                for (size_t f = 0; f < FEATURE_DIM; ++f) {
                    double diff = features_test[i][f] - features_train[j][f];
                    sum += diff * diff;
                }

                curr_dist.push_back({sqrt(sum), labels_train[j]});
            }

            double iter_dist_end = omp_get_wtime();
            double dist_time = iter_dist_end - iter_dist_start;

            // Sort distances to find k nearest neighbors
            double start_sort = omp_get_wtime();

            // Use parallel quicksort with tasks
            #pragma omp parallel num_threads(1)
            {
                #pragma omp single
                {
                    quick_sort_parallel(curr_dist, 0, curr_dist.size() - 1);
                }
            }

            double end_sort = omp_get_wtime();
            double sort_time = end_sort - start_sort;

            // Track timing information
            thread_timings.push_back({dist_time, sort_time});

            // Find majority label among k nearest neighbors
            int predicted_label = find_majority_label(curr_dist, k);
            all_predictions[i] = predicted_label;

            // Update progress counter
            thread_progress++;
            
            // Atomically update the global progress counter
            #pragma omp atomic
            total_processed++;
            
            // Report progress occasionally (only from thread 0 to avoid mixed output)
            if (omp_get_thread_num() == 0 && (i % (NUM_TEST_SAMPLES/10) == 0 || i == NUM_TEST_SAMPLES-1)) {
                #pragma omp critical
                {
                    cout << "  Processed approximately " << total_processed << "/" 
                         << NUM_TEST_SAMPLES << " (" 
                         << (total_processed * 100 / NUM_TEST_SAMPLES) 
                         << "%) samples" << endl;
                }
            }
        }

        // Aggregate timing information from all threads
        #pragma omp critical
        {
            for (const auto& timing : thread_timings) {
                total_dist_time_acc += timing.first;
                total_sort_time_acc += timing.second;
            }
        }
    }

// Calculate and report timing information
double end_overall_time = omp_get_wtime();
double total_runtime = end_overall_time - start_overall_time;

// Calculate average time per thread
int actual_thread_count = omp_get_max_threads();
double avg_dist_time = total_dist_time_acc / actual_thread_count;
double avg_sort_time = total_sort_time_acc / actual_thread_count;

cout << "  Finished Parallel Processing for K=" << k << endl;
cout << "  Avg Distance Calc Time per Thread: " << avg_dist_time << " s" << endl;
cout << "  Avg Sorting Time per Thread:       " << avg_sort_time << " s" << endl;
cout << "  Total Runtime:                     " << total_runtime << " s" << endl;

return all_predictions;
}

void quick_sort(vector<distance_single_sample> &vec, int low, int high) {
    if (low < high) {
        int pi = partition(vec, low, high);
        quick_sort(vec, low, pi - 1);
        quick_sort(vec, pi + 1, high);
    }
}

vector<int> run_knn_serial(vector<vector<float>>& features_train, vector<int>& labels_train, vector<vector<float>>& features_test, size_t NUM_SAMPLES_TRAIN, size_t FEATURE_DIM, size_t NUM_TEST_SAMPLES, int k) {
    vector<int> all_predictions(NUM_TEST_SAMPLES);

    cout << "Processing KNN for K=" << k << "..." << endl;

    double total_dist_time_acc = 0.0;
    double total_sort_time_acc = 0.0;
    double start_overall_time = omp_get_wtime();
    
    size_t processed_count = 0;
    size_t report_interval = NUM_TEST_SAMPLES / 10;
    if (report_interval == 0) {
        report_interval = 1;
    }

    for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
        vector<distance_single_sample> curr_dist;
        curr_dist.reserve(NUM_SAMPLES_TRAIN);

        double iter_dist_start = omp_get_wtime();
        for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < FEATURE_DIM; ++k) {
                double diff = features_test[i][k] - features_train[j][k];
                diff *= diff;
                sum += diff;
            }
            double eu_dist_sample = sqrt(sum);
            curr_dist.push_back({eu_dist_sample, labels_train[j]});
        }
        double iter_dist_end = omp_get_wtime();
        total_dist_time_acc += (iter_dist_end - iter_dist_start);

        double start_sort = omp_get_wtime();
        if (!curr_dist.empty()) {
            quick_sort(curr_dist, 0, curr_dist.size() - 1);
        }
        double end_sort = omp_get_wtime();
        total_sort_time_acc += (end_sort - start_sort);

        int predicted_label = find_majority_label(curr_dist, k);
        all_predictions[i] = predicted_label;

        processed_count++;
        if (processed_count % report_interval == 0) {
            cout << "  Processed " << processed_count << "/" 
                 << NUM_TEST_SAMPLES << " (" 
                 << (processed_count * 100 / NUM_TEST_SAMPLES) 
                 << "%) samples" << endl;
        }
    }

    double end_overall_time = omp_get_wtime();
    double total_runtime = end_overall_time - start_overall_time;

    cout << "  Finished Processing for K=" << k << endl;
    cout << "  Distance Calc Time: " << total_dist_time_acc << " s" << endl;
    cout << "  Sorting Time:       " << total_sort_time_acc << " s" << endl;
    cout << "  Total Runtime:      " << total_runtime << " s" << endl;

    return all_predictions;
}

int main() {
    // Dataset dimensions
    constexpr size_t NUM_SAMPLES_TRAIN = 50000;  // Number of training samples
    constexpr size_t FEATURE_DIM = 512;          // Feature dimension
    constexpr size_t NUM_TEST_SAMPLES = 10000;   // Number of test samples
    
    cout << "Loading training samples... \n" << endl;
    // Load training data
    vector<vector<float>> features_train = read_features("train/train_features.bin", NUM_SAMPLES_TRAIN, FEATURE_DIM);
    vector<int> labels_train = read_labels("train/train_labels.bin", NUM_SAMPLES_TRAIN);
    
    cout << "Loading test samples... \n" << endl;
    // Load test data
    vector<vector<float>> features_test = read_features("test/test_features.bin", NUM_TEST_SAMPLES, FEATURE_DIM);
    vector<int> labels_test = read_labels("test/test_labels.bin", NUM_TEST_SAMPLES);

    // Different values of k to test
    const vector<int> K_Vals = {3, 5, 7};
    
    // Determine thread counts to test
    int max_threads = omp_get_max_threads();
    vector<int> thread_counts = {1, max_threads/2, max_threads};
    
    cout << "Maximum available threads: " << max_threads << endl;
    cout << "==============================================" << endl;

    // First run the serial version for baseline
    cout << "\n=== Running Serial KNN ===" << endl;
    vector<double> serial_times;
    vector<double> serial_accuracies;
    
    for (int k : K_Vals) {
        double start_time = omp_get_wtime();
        vector<int> predicted_labels = run_knn_serial(features_train, labels_train, features_test, 
                                                     NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES, k);
        double end_time = omp_get_wtime();
        double runtime = end_time - start_time;
        serial_times.push_back(runtime);
        
        // Calculate accuracy
        int correct_count = 0;
        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
            if (predicted_labels[i] == labels_test[i]) {
                correct_count++;
            }
        }
        double accuracy = (NUM_TEST_SAMPLES > 0) 
                        ? static_cast<double>(correct_count) / NUM_TEST_SAMPLES * 100.0
                        : 0.0;
        
        serial_accuracies.push_back(accuracy);
        cout << "Serial KNN (K=" << k << ") - Accuracy: " << accuracy << "%, Runtime: " << runtime << " s" << endl;
        cout << "---------------------------------------------" << endl;
    }

    // Run parallel versions with different thread counts
    cout << "\n=== Running Parallel KNN ===" << endl;
    
    for (int num_threads : thread_counts) {
        cout << "\nParallel KNN with " << num_threads << " threads:" << endl;
        
        for (size_t k_idx = 0; k_idx < K_Vals.size(); ++k_idx) {
            int k = K_Vals[k_idx];
            
            double start_time = omp_get_wtime();
            vector<int> predicted_labels = run_knn_parallel(features_train, labels_train, features_test, 
                                                          NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES, 
                                                          k, num_threads);
            double end_time = omp_get_wtime();
            double runtime = end_time - start_time;
            
            // Calculate accuracy
            int correct_count = 0;
            for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                if (predicted_labels[i] == labels_test[i]) {
                    correct_count++;
                }
            }
            double accuracy = (NUM_TEST_SAMPLES > 0) 
                            ? static_cast<double>(correct_count) / NUM_TEST_SAMPLES * 100.0
                            : 0.0;
            
            // Calculate speedup relative to serial version
            double speedup = serial_times[k_idx] / runtime;
            
            cout << "Parallel KNN (K=" << k << ", " << num_threads << " threads):" << endl;
            cout << "  Accuracy: " << accuracy << "%" << endl;
            cout << "  Runtime: " << runtime << " s" << endl;
            cout << "  Speedup: " << speedup << "x over serial version" << endl;
            cout << "  Parallel Efficiency: " << (speedup / num_threads) * 100 << "%" << endl;
            
            // Verify results match
            bool results_match = true;
            vector<int> serial_labels = run_knn_serial(features_train, labels_train, features_test, 
                                                      NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES, k);
            
            for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                if (predicted_labels[i] != serial_labels[i]) {
                    results_match = false;
                    break;
                }
            }
            
            cout << "  Results match serial version: " << (results_match ? "Yes" : "No") << endl;
            cout << "---------------------------------------------" << endl;
        }
    }
    
    cout << "\n=== Performance Summary ===" << endl;
    cout << "Serial runtime for different K values:" << endl;
    
    for (size_t i = 0; i < K_Vals.size(); ++i) {
        cout << "  K=" << K_Vals[i] << ": " << serial_times[i] << " s (Accuracy: " << serial_accuracies[i] << "%)" << endl;
    }
    
    cout << "\nFor best parallel performance, consider:" << endl;
    cout << "1. Increasing thread count based on available CPU cores" << endl;
    cout << "2. Optimizing memory access patterns" << endl;
    cout << "3. Adjusting the chunk size in the dynamic schedule" << endl;
    
    return 0;
}