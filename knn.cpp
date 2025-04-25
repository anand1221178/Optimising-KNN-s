#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>
#include <cmath>
#include <utility>
#include <algorithm>

using namespace std;

struct distance_single_sample
{
    double distance;
    int label;
};


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

    // If k is 0 or less, or no distances, return a default/error label
    // if (num_neighbors_to_check <= 0) {
    //     return -1; // Or another indicator that prediction failed
    // }
    
    for (int i = 0; i < num_neighbors_to_check; ++i) {
        int label = sorted_distances[i].label;
        // Check if the label is within the expected range (0-9)
        if (label >= 0 && label < num_classes) {
            votes[label]++;
        }
    }

    // Find the label with the maximum votes
    int predicted_label = 0; // Default prediction (e.g., class 0)
    int max_votes = -1;

    for (int label_index = 0; label_index < num_classes; ++label_index) {
        if (votes[label_index] > max_votes) {
            max_votes = votes[label_index];
            predicted_label = label_index;
        }
        // Basic tie-breaking: the first label encountered with max_votes wins.
    }

    return predicted_label;
}

int partition(vector<distance_single_sample> &vec, int low, int high) {

    // Check basic bounds upfront (optional but safer)
    if (low >= high || low < 0 || high >= vec.size()) {
        return low; // Or handle error appropriately
    }

    // Selecting the distance of the last element as the pivot value
    double pivot_distance = vec[high].distance; // Correctly get the distance

    // Index of the smaller element
    int i = (low - 1);

    for (int j = low; j < high; j++) { // Loop up to high-1

        // If current element's distance is smaller than the pivot distance
        // Using '<' is more standard for Lomuto partition to avoid issues with all equal elements
        if (vec[j].distance < pivot_distance) { // Correctly compare distances
            i++;
            swap(vec[i], vec[j]); // Swap the whole structs
        }
    }

    // Put the pivot element (vec[high]) into its correct sorted position
    swap(vec[i + 1], vec[high]);

    // Return the partition index (the pivot's final position)
    return (i + 1);
}

void quick_sort(vector<distance_single_sample> &vec, int low, int high)
{
    // Base case: This part will be executed till the starting
    // index low is lesser than the ending index high
    if (low < high) {

        // pi is Partitioning Index, arr[p] is now at
        // right place
        int pi = partition(vec, low, high);

        // Separately sort elements before and after the
        // Partition Index pi
        #pragma omp tasks shared(vec)
        quick_sort(vec, low, pi - 1);
        #pragma omp tasks shared(vec)
        quick_sort(vec, pi + 1, high);
    }
}

vector<int> run_knn_serial(vector<vector<float>>& features_train, vector<int>& labels_train, vector<vector<float>>& features_test, size_t NUM_SAMPLES_TRAIN, size_t FEATURE_DIM, size_t NUM_TEST_SAMPLES, int k)
{
    // Find the distances of test samples against training samples
    //Have to iterate over each sample and send in the features of each sample, not the actual sample!
    vector<int> all_predictions;
    all_predictions.reserve(NUM_TEST_SAMPLES);

    cout << "Processing KNN for K=" << k << "..." << endl;

    //Set timing 
    double total_dist_time = 0.0;
    double total_sort_time = 0.0;

    // Start overall time here
    double start_overall_time = omp_get_wtime();


    //reduction(+:total_dist_time_acc, total_sort_time_acc)
    #pragma omp parallel for  schedule(static) num_threads(16)
    //Loop over test samples -> since we are comparing that to the train samples 
    for(size_t i = 0; i < NUM_TEST_SAMPLES; ++i)
    {
        // Define distance vector of struct we made to store sample distances and labels
        vector<distance_single_sample> curr_dist;
        //Reserve space
        curr_dist.reserve(NUM_SAMPLES_TRAIN);

        // Loop over each training sample since we compare each test sample to each train sample
        for (size_t j = 0; j < NUM_SAMPLES_TRAIN; ++j)
        {
            //Sum for this sample
            double sum = 0.0;
            //Loop over each feature in the test sample and train sample and get euclidean distance of them
            for (size_t k = 0; k < FEATURE_DIM; ++k)
            {
                // Distance of single feature from train to test in a SINGLE SAMPLE
                double diff = features_test[i][k] - features_train[j][k];
                diff *= diff;
                sum += diff;
            } //end k loop
            double eu_dist_sample = sqrt(sum);
            // Store distance in struct
            curr_dist.push_back({eu_dist_sample, labels_train[j]});
        } //end j loop

        // SORT SINGLE SAMPLE
        double start_sort = omp_get_wtime();
        if(!curr_dist.empty())
        {
            quick_sort(curr_dist, 0, curr_dist.size() -1);
        }
        double end_sort = omp_get_wtime();
        total_sort_time += (end_sort - start_sort);

        // PREDICTION OF A SINGLE SAMPLE
        int predicted_label = find_majority_label(curr_dist, k);
        all_predictions.push_back(predicted_label);


    //     // COOL LENGTH THINGY
    //     if ((i + 1) % 100 == 0) {
    //         cout << "  Processed " << (i + 1) << "/" << NUM_TEST_SAMPLES << " test samples for K=" << k << endl;
    //    }
    }//end i loop

    double end_overall_time = omp_get_wtime();
    double total_runtime = end_overall_time - start_overall_time;


    cout << "  Finished Processing for K=" << k << endl;
    cout << "  Distance Calc Time: " << total_dist_time << " s" << endl;
    cout << "  Sorting Time:       " << total_sort_time << " s" << endl;
    cout << "  Total Runtime:      " << total_runtime << " s" << endl;


    return all_predictions; // Return the vector of predictions

}//end function


int main() {

    constexpr size_t NUM_SAMPLES_TRAIN = 50000;
    constexpr size_t FEATURE_DIM = 512;
    constexpr size_t NUM_TEST_SAMPLES = 10000;
    
    // Training Samples
    cout << "Loading training samples... \n" << endl;
    vector<vector<float>> features_train = read_features("train/train_features.bin", NUM_SAMPLES_TRAIN, FEATURE_DIM);
    vector<int> labels_train = read_labels("train/train_labels.bin", NUM_SAMPLES_TRAIN);
    
    //Test Samples
    cout << "Loading test samples... \n" << endl;
    vector<vector<float>> features_test = read_features("test/test_features.bin", NUM_TEST_SAMPLES, FEATURE_DIM);
    vector<int> labels_test  = read_labels("test/test_labels.bin", NUM_TEST_SAMPLES);

    // store k in vec, just casue
    const vector<int> K_Vals = {3,5,7};


    for (int k : K_Vals)
    {
        
        vector<int> predicted_labels = run_knn_serial(features_train, labels_train,features_test,NUM_SAMPLES_TRAIN, FEATURE_DIM, NUM_TEST_SAMPLES, k);

        
        int correct_count = 0;
        
        if (predicted_labels.size() == labels_test.size()) {
            for (size_t i = 0; i < NUM_TEST_SAMPLES; ++i) {
                if (predicted_labels[i] == labels_test[i]) {
                    correct_count++;
                }
            }
        }

        double accuracy = (NUM_TEST_SAMPLES > 0)
                          ? static_cast<double>(correct_count) / NUM_TEST_SAMPLES
                          : 0.0;

        cout << "  Accuracy for K = " << k << ": " << accuracy * 100.0 << "%" << endl;
        cout << "------------------------------------" << endl;

    }


    return 0;
}

