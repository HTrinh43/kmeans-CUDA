#include <io.h>

void read_file(struct options_t* args,
               int*              n_vals,
               std::vector<Point>& dataSet,
               Point_cu& dataSet2) {

  	// Open file
	std::ifstream in;
	in.open(args->in_file);
	// Get num vals
	in >> *n_vals;
	int n_dim;
	n_dim = args->d;
    float* points = new float[*n_vals*n_dim];
    int* labels = new int[*n_vals];
	// Read input vals
    dataSet2.size = *n_vals;
    for (int i = 0; i < *n_vals; i++) {
        int index;
		Point pt;
        in >> index;
        pt.label = -1;
        labels[i] = -1;

        
        for (int j = 0; j < n_dim; j++) {
            int indx = i * n_dim + j;
			double val;
            in >> val;
            pt.features.push_back(val);
            points[indx] = static_cast<float>(val);
        }
		dataSet.push_back(pt);
    }
    dataSet2.features = points;
    dataSet2.label = labels;

	in.close();
}
