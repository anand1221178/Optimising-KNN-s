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
    //Set number of target classes
    const int num_classes = 10; 

    //Create vector votes with 10 rows for each class each initiallised to 0
    vector<int> votes(num_classes, 0);
    
    int num_neighbors_to_check = min(k, static_cast<int>(sorted_distances.size()));

    //Accumalte number of votes for each label into vec votes
    for (int i = 0; i < num_neighbors_to_check; ++i) {
        int label = sorted_distances[i].label;
        if (label >= 0 && label < num_classes) {
            votes[label]++;
        }
    }

    // initial setting
    int predicted_label = 0;
    int max_votes = -1;


    //Find majority label
    for (int label_index = 0; label_index < num_classes; ++label_index) {
        if (votes[label_index] > max_votes) {
            max_votes = votes[label_index];
            predicted_label = label_index;
        }
    }

    return predicted_label;
}

// Partition for Qsort
int partition(vector<distance_single_sample> &vec, int low, int high) {
    
    // Edge case check -> return low if not in range
    if (low >= high || low < 0 || high >= static_cast<int>(vec.size())) {
        return low;
    }

    // Last element set as pivot
    double pivot_distance = vec[high].distance;
    int i = (low - 1);

    // Rearrange elements -> smaller b4 pivot
    for (int j = low; j < high; j++) {
        if (vec[j].distance < pivot_distance) {
            i++;
            swap(vec[i], vec[j]);
        }
    }

    // Place pivot correctly
    swap(vec[i + 1], vec[high]);
    return (i + 1);
}

// OMP task Qsort implentation
void quick_sort_parallel(vector<distance_single_sample> &vec, int low, int high, int depth = 0) {
    
    // Limit task creation depth
    const int MAX_TASK_DEPTH = 3;
    
    if (low < high) {
        // Find partition index
        int pi = partition(vec, low, high);
        
        if (depth >= MAX_TASK_DEPTH) {
            // Sequential execution for deeper recursion levels
            quick_sort(vec, low, pi - 1);
            quick_sort(vec, pi + 1, high);
        } else {
            // Parallel execution using tasks for shallow recursion
                #pragma omp task firstprivate(low,pi,depth) shared(vec)
                quick_sort_parallel(vec, low, pi - 1, depth + 1);
                #pragma omp task firstprivate(low,pi,depth) shared(vec)
                quick_sort_parallel(vec, pi + 1, high, depth + 1);

                // Wait for both tasks to complete before moving on
                #pragma omp taskwait

        }
    }
}

// QSort par w sections
void quick_sort_parallel_sections(vector<distance_single_sample> &vec, int low, int high, int depth = 0) {
    const int MAX_TASK_DEPTH = 3;// Limit recursion depth for parallel sections

    if (low < high) {
        int pi = partition(vec, low, high);

        if(depth >= MAX_TASK_DEPTH)
        {
            // For deep recursion, fall back to sequential quicksort
            quick_sort(vec, low, pi - 1);
            quick_sort(vec, pi + 1, high);
        }
        else{
             // Use OpenMP parallel sections to sort left and right partitions concurrently
            #pragma omp parallel sections
            {
                #pragma omp section
                quick_sort(vec, low, pi - 1);
                #pragma omp section
                quick_sort(vec, pi + 1, high);
            }
        }

    }
}

vector<int> run_knn_parallel_sections(vector<vector<float>>& features_train, vector<int>& labels_train, vector<vector<float>>& features_test, size_t NUM_SAMPLES_TRAIN, size_t FEATURE_DIM, size_t NUM_TEST_SAMPLES, int k, int num_threads)
{
    vector<int> all_predictions(NUM_TEST_SAMPLES);
    omp_set_num_threads(num_threads);

    double total_dist_time = 0.0;
    double total_sort_time = 0.0;
    double t_start = omp_get_wtime();


    // per‐thread accumulators
    double dist_time_acc = 0.0;
    double sort_time_acc = 0.0;
    
    // int split = int(NUM_SAMPLES_TRAIN / num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
            // --- distance calculation ---
            //Struct of dist and label
            vector<distance_single_sample> curr_dist(NUM_SAMPLES_TRAIN);
            double td0 = omp_get_wtime();
            for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j) { //compare each test sample to each train sample
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < FEATURE_DIM; ++k) {
                    double diff = features_test[i][k] - features_train[j][k];
                    sum += diff * diff;
                }
                    curr_dist[j] = { sqrt(sum), labels_train[j] };
            }

            double td1 = omp_get_wtime();
            #pragma omp atomic
            dist_time_acc += (td1 - td0);

            // Distance for each feature in a SINGLE sample now found

            // --- task‐based quicksort ---
            double ts0 = omp_get_wtime();
            quick_sort_parallel_sections(curr_dist, 0, curr_dist.size()-1);
            double ts1 = omp_get_wtime();

            // now the vector is fully sorted
            #pragma omp atomic
            sort_time_acc += (ts1 - ts0);
            
            
            all_predictions[i] = find_majority_label(curr_dist, k);
        } //end I

    }// reduce back to global totals

    total_dist_time += dist_time_acc;

    total_sort_time += sort_time_acc;
    
    double t_end = omp_get_wtime();
    std::cout << "  Finished Parallel Processing for K=" << k << std::endl;
    std::cout << "  Total Runtime:                     " << (t_end - t_start) << " s" << std::endl;
    std::cout << "  Avg Distance Calc Time per Thread: "
         << (total_dist_time / omp_get_max_threads()) << " s" << std::endl;
    std::cout << "  Avg Sorting Time per Thread:       "
         << (total_sort_time / omp_get_max_threads()) << " s" << endl;

    return all_predictions;
}

// Parallel KNN func
vector<int> run_knn_parallel(vector<vector<float>>& features_train, vector<int>& labels_train, vector<vector<float>>& features_test, size_t NUM_SAMPLES_TRAIN, size_t FEATURE_DIM, size_t NUM_TEST_SAMPLES, int k, int num_threads)
{
    vector<int> all_predictions(NUM_TEST_SAMPLES);
    omp_set_num_threads(num_threads);

    double total_dist_time = 0.0;
    double total_sort_time = 0.0;
    double t_start = omp_get_wtime();


    // per‐thread accumulators
    double dist_time_acc = 0.0;
    double sort_time_acc = 0.0;
    
    // int split = int(NUM_SAMPLES_TRAIN / num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
            // --- distance calculation ---
            //Struct of dist and label
            vector<distance_single_sample> curr_dist(NUM_SAMPLES_TRAIN);
            double td0 = omp_get_wtime();
            for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j) { //compare each test sample to each train sample
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < FEATURE_DIM; ++k) {
                    double diff = features_test[i][k] - features_train[j][k];
                    sum += diff * diff;
                }
                    curr_dist[j] = { sqrt(sum), labels_train[j] };
            }

            double td1 = omp_get_wtime();
            #pragma omp atomic
            dist_time_acc += (td1 - td0);

            // Distance for each feature in a SINGLE sample now found

            // --- task‐based quicksort ---
            double ts0 = omp_get_wtime();
            quick_sort_parallel(curr_dist, 0, curr_dist.size()-1);
            double ts1 = omp_get_wtime();

            // now the vector is fully sorted
            #pragma omp atomic
            sort_time_acc += (ts1 - ts0);
            
            
            all_predictions[i] = find_majority_label(curr_dist, k);
        } //end I

    }// reduce back to global totals

    total_dist_time += dist_time_acc;

    total_sort_time += sort_time_acc;
    
    double t_end = omp_get_wtime();
    std::cout << "  Finished Parallel Processing for K=" << k << std::endl;
    std::cout << "  Total Runtime:                     " << (t_end - t_start) << " s" << std::endl;
    std::cout << "  Avg Distance Calc Time per Thread: "
         << (total_dist_time / omp_get_max_threads()) << " s" << std::endl;
    std::cout << "  Avg Sorting Time per Thread:       "
         << (total_sort_time / omp_get_max_threads()) << " s" << endl;

    return all_predictions;
}


void quick_sort(vector<distance_single_sample> &vec, int low, int high) {  
    if (low < high) {
        int pi = partition(vec, low, high);
        quick_sort(vec, low, pi - 1);
        quick_sort(vec, pi + 1, high);
    }
}

// Serial implementation of the K-Nearest Neighbors (KNN) classifier
vector<int> run_knn_serial(vector<vector<float>>& features_train, vector<int>& labels_train,vector<vector<float>>& features_test, size_t NUM_SAMPLES_TRAIN,size_t FEATURE_DIM, size_t NUM_TEST_SAMPLES, int k) {
    vector<int> all_predictions(NUM_TEST_SAMPLES);  // Stores predicted labels for each test sample

    cout << "Processing KNN for K=" << k << "..." << endl;

    double total_dist_time_acc = 0.0;
    double total_sort_time_acc = 0.0;
    double start_overall_time = omp_get_wtime();  // Start timer for entire run

    size_t processed_count = 0;
    size_t report_interval = NUM_TEST_SAMPLES / 10;
    if (report_interval == 0) report_interval = 1;  // Ensure we report at least once

    // Iterate over all test samples
    for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
        vector<distance_single_sample> curr_dist;
        curr_dist.reserve(NUM_SAMPLES_TRAIN);  // Preallocate for efficiency

        // --- Distance Calculation ---
        double iter_dist_start = omp_get_wtime();
        for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j) {
            double sum = 0.0;
            // Euclidean distance calculation
            for (size_t k = 0; k < FEATURE_DIM; ++k) {
                double diff = features_test[i][k] - features_train[j][k];
                sum += diff * diff;
            }
            double eu_dist_sample = sqrt(sum);
            curr_dist.push_back({eu_dist_sample, labels_train[j]});
        }
        double iter_dist_end = omp_get_wtime();
        total_dist_time_acc += (iter_dist_end - iter_dist_start);

        // --- Sorting distances ---
        double start_sort = omp_get_wtime();
        if (!curr_dist.empty()) {
            quick_sort(curr_dist, 0, curr_dist.size() - 1);  // Serial quicksort
        }
        double end_sort = omp_get_wtime();
        total_sort_time_acc += (end_sort - start_sort);

        // --- Find predicted label based on K closest distances ---
        int predicted_label = find_majority_label(curr_dist, k);
        all_predictions[i] = predicted_label;

        // Progress reporting
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

    // --- Reporting timings ---
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
    
    cout << "Maximum available threads: " << max_threads << endl;
    cout << "==============================================" << endl;

    // First run the serial version for baseline
    cout << "\n=== Running Serial KNN ===" << endl;
    vector<double> serial_times;
    vector<double> serial_accuracies;
    vector<vector<int>> all_serial_predictions;
    
    for (int k : K_Vals) {
        double total_runtime = 0.0;
        double total_accuracy = 0.0;
        vector<int> serial_predictions;

        for (int run = 0; run < 10; ++run) {
            double start_time = omp_get_wtime();
            vector<int> predictions = run_knn_serial(features_train, labels_train, features_test,
                                                    NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES, k);
            double end_time = omp_get_wtime();
            total_runtime += (end_time - start_time);

            int correct = 0;
            for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                if (predictions[i] == labels_test[i]) correct++;
            }
            total_accuracy += 100.0 * correct / NUM_TEST_SAMPLES;

            if (run == 0) serial_predictions = predictions;
        }

    double avg_runtime = total_runtime / 10.0;
    double avg_accuracy = total_accuracy / 10.0;

    serial_times.push_back(avg_runtime);
    serial_accuracies.push_back(avg_accuracy);
    all_serial_predictions.push_back(serial_predictions);

    cout << "Serial KNN (K=" << k << ") - Avg Accuracy: " << avg_accuracy
        << "%, Avg Runtime: " << avg_runtime << " s" << endl;

    }

    // Run parallel versions with different thread counts
        ofstream outfile("knn_scaling_results.csv");
        outfile << "k,num_threads,runtime,speedup,efficiency,accuracy\n";
        vector<int> thread_counts = {1, 2, 4, 8, 16, 32, 64, 128};

        for (int k_idx = 0; k_idx < K_Vals.size(); ++k_idx) {
            int k = K_Vals[k_idx];
            const double serial_runtime = serial_times[k_idx];
            const double serial_accuracy = serial_accuracies[k_idx];
            const vector<int>& serial_predictions = all_serial_predictions[k_idx];

            // Thread counts to test (no NUM_TEST_SAMPLES scaling)

            for (int num_threads : thread_counts) {
                omp_set_num_threads(num_threads);

                double start_time = omp_get_wtime();
                vector<int> predicted_labels = run_knn_parallel(features_train, labels_train, features_test,
                                                                NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES,
                                                                k, num_threads);
                double end_time = omp_get_wtime();
                double runtime = end_time - start_time;

                int correct = 0;
                bool match = true;
                for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                    if (predicted_labels[i] == labels_test[i]) correct++;
                    if (predicted_labels[i] != serial_predictions[i]) match = false;
                }

                double accuracy = 100.0 * correct / NUM_TEST_SAMPLES;
                double speedup = serial_runtime / runtime;
                double efficiency = (speedup / num_threads) * 100.0;

                // Console summary
                cout << "K=" << k << ", Threads=" << num_threads
                    << " | Time=" << runtime << " s, Speedup=" << speedup
                    << "x, Efficiency=" << efficiency << "%, Accuracy=" << accuracy
                    << "% -> " << (match ? " Match" : "Mismatch") << endl;

                // CSV file output
                outfile << k << "," << num_threads << "," << runtime << ","
                        << speedup << "," << efficiency << "," << accuracy << "\n";
            }

            cout << "---------------------------------------------" << endl;
        }

        outfile.close();

        ofstream section_out("knn_sections_vs_tasks.csv");
        section_out << "k,num_threads,method,runtime,speedup,efficiency,accuracy\n";

        for (int k_idx = 0; k_idx < K_Vals.size(); ++k_idx) {
            int k = K_Vals[k_idx];
            const double serial_runtime = serial_times[k_idx];
            const vector<int>& serial_predictions = all_serial_predictions[k_idx];

            for (int num_threads : thread_counts) {
                // TASK-based run
                {
                    omp_set_num_threads(num_threads);
                    double task_runtime = 0.0;
                    double task_accuracy = 0.0;

                    for (int run = 0; run < 10; ++run) {
                        omp_set_num_threads(num_threads);
                        double start = omp_get_wtime();
                        vector<int> task_preds = run_knn_parallel(features_train, labels_train, features_test,
                                                                NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES,
                                                                k, num_threads);
                        double end = omp_get_wtime();
                        task_runtime += (end - start);

                        int correct = 0;
                        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                            if (task_preds[i] == labels_test[i]) correct++;
                        }

                        task_accuracy += 100.0 * correct / NUM_TEST_SAMPLES;
                    }

                    task_runtime /= 10.0;
                    task_accuracy /= 10.0;
                    double task_speedup = serial_runtime / task_runtime;
                    double task_efficiency = (task_speedup / num_threads) * 100.0;

                    section_out << k << "," << num_threads << ",task,"
                                << task_runtime << "," << task_speedup << "," << task_efficiency << "," << task_accuracy << "\n";

                    cout << "TASK     | K=" << k << ", Threads=" << num_threads
                        << ", Runtime=" << task_runtime << "s, Speedup=" << task_speedup
                        << "x, Eff=" << task_efficiency << "%, Acc=" << task_accuracy << "%" << endl;


                }

                // SECTIONS-based run
                {
                    omp_set_num_threads(num_threads);
                    double sec_runtime = 0.0;
                    double sec_accuracy = 0.0;

                    for (int run = 0; run < 10; ++run) {
                        omp_set_num_threads(num_threads);
                        double start = omp_get_wtime();
                        vector<int> section_preds = run_knn_parallel_sections(features_train, labels_train, features_test,
                                                                            NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES,
                                                                            k, num_threads);
                        double end = omp_get_wtime();
                        sec_runtime += (end - start);

                        int correct = 0;
                        for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                            if (section_preds[i] == labels_test[i]) correct++;
                        }

                        sec_accuracy += 100.0 * correct / NUM_TEST_SAMPLES;
                    }

                    sec_runtime /= 10.0;
                    sec_accuracy /= 10.0;
                    double sec_speedup = serial_runtime / sec_runtime;
                    double sec_efficiency = (sec_speedup / num_threads) * 100.0;

                    section_out << k << "," << num_threads << ",sections,"
                                << sec_runtime << "," << sec_speedup << "," << sec_efficiency << "," << sec_accuracy << "\n";

                    cout << "SECTIONS | K=" << k << ", Threads=" << num_threads
                        << ", Runtime=" << sec_runtime << "s, Speedup=" << sec_speedup
                        << "x, Eff=" << sec_efficiency << "%, Acc=" << sec_accuracy << "%" << endl;


                }
            }

            cout << "---------------------------------------------" << endl;
        }
        section_out.close();



    
    cout << "\n=== Performance Summary ===" << endl;
    cout << "Serial runtime for different K values:" << endl;
    
    for (size_t i = 0; i < K_Vals.size(); ++i) {
        cout << "  K=" << K_Vals[i] << ": " << serial_times[i] << " s (Accuracy: " << serial_accuracies[i] << "%)" << endl;
    }

    
    return 0;
}
