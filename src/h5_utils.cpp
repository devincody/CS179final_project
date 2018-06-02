/**
 * Implementation of h5 dataset decoder
 * @author Devin Cody
 * @date June 1, 2018
 */


#include "h5_utils.hpp"

// little test script
int tests(){
	float ans_weight[10] = {0.42947057,0.373467,-0.061360113,0.27476987,0.038680777,-0.36722335,-0.05746817,-0.26224968,-0.35009676,0.55037946};
	float ans_bias[10] = {0.73429835,0.093403675,0.06775674,0.8862966,0.25994542,0.66426694,-0.015828926,0.3249065,0.68600726,0.062479325};
	// float ans_weight[10] = {0.341195, 0.339992, -0.0448442, 0.232159, 0.0897821, -0.303314, -0.0726092, -0.246675, -0.336833, 0.464184};
	// float ans_bias[10] = {0.7301776,0.06493629,0.03428847,0.8260386,0.2578029,0.54867655,-0.012438543,0.34789944,0.5510871,0.06297145};
	float tolerance = 1E-5;

	std::cout << "***********************************" <<std::endl;
	std::cout << "*         Reading Weights         *" <<std::endl;
	std::cout << "***********************************" <<std::endl;
	float *data = new float[64*3*3*3]();
	get_weights("block1_conv1", 64, 3, 3, 3, data);

	int check = 0;
	for (int i = 0; i < 10; i ++){
		if (std::abs(data[i] - ans_weight[i]) > tolerance){
			check = 1;
		}
		std::cout << "weights["<<i<<"] = " << data[i] << std::endl;
	}

	if (check){
		std::cout << "error reading file" << std::endl;
	} else {
		std::cout << "File read correctly" << std::endl;
	}

	delete[] data;

	std::cout << "\n***********************************" <<std::endl;
	std::cout << "*         Reading Biases          *" <<std::endl;
	std::cout << "***********************************" <<std::endl;

	data = new float[64]();
	get_bias("block1_conv1", 64, data);
	
	check = 0;
	for (int i = 0; i < 10; i ++){
		if (std::abs(data[i] - ans_bias[i]) > tolerance){
			check = 1;
		}
		std::cout << "bias["<<i<<"] = " << data[i] << std::endl;
	}

	if (check){
		std::cout << "error reading file" << std::endl;
	} else {
		std::cout << "File read correctly" << std::endl;
	}

	delete[] data;
	return 0;
}

int get_weights(std::string name, int n, int c, int h, int w, float* odata){
	int len = n*c*h*w;

	float* data = new float[len]();
	// float* odata = new float[len]();


	memset(data, 0, len*sizeof(float));
	// memset(odata, 0, len*sizeof(float));

	// std::cout << "done weights memsetting for: "<< name << std::endl;
	std::cout <<  "get_weights = n: " << n << ", c: " << c << ", h: "<< h << ", w:" << w << std::endl;

	try{
		H5::Exception::dontPrint();
		H5::H5File file(FILE_NAME, H5F_ACC_RDONLY); // open read only file
		H5::Group group = file.openGroup(name);		// open file group
		group = group.openGroup(name);
		H5::DataSet dataset = group.openDataSet(kernel_NAME); // open dataset

	    H5T_class_t type_class = dataset.getTypeClass(); 	  // check datatype

	    if (type_class != H5T_FLOAT){
	    	std::cout <<"ERROR: Not float type in " << name << std::endl;
	    }		

	    dataset.read(data, H5::PredType::NATIVE_FLOAT);
	    // std::cout << "done weights Reading " << std::endl;

	    // Format data into NCHW from HWCN 
	    if (data){
			int z = 0;
			int loc = 0;
			for(int i = 0; i < n; ++i){
				for(int j = 0; j < c; ++j){
					for(int k = 0; k < h; ++k){
						for(int l = 0; l < w; ++l){
							loc = i + j*n + l*n*c + k*n*c*w; // OG
							// loc = i + j*n + k*n*c + l*n*c*w;
							odata[z] = data[loc];
							z++;
						}
					}
				}
			}
		} else {
			std::cout << "ERROR: Failed to read weight data" << std::endl;
		}

		// std::cout << "done weights reordering " << std::endl;

		file.close();


	}
	// catch failure caused by the H5File operations
	catch( H5::FileIException error )
	{
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		error.printError();
		return 1;
	}

	delete[] data;
	return 0;
}





int get_bias(std::string name, int n, float* odata){
	// float* odata = new float[n]();

	// std::cout << "done bias memsetting for: " << name << std::endl;
	std::cout <<  "get bias = n: " << n << std::endl;
	// memset(odata, 0, n*sizeof(float));

	try{
		H5::Exception::dontPrint();
		H5::H5File file(FILE_NAME, H5F_ACC_RDONLY); // open read only file
		H5::Group group = file.openGroup(name);		// open file group
		group = group.openGroup(name);
		H5::DataSet dataset = group.openDataSet(bias_NAME); // open dataset

	    H5T_class_t type_class = dataset.getTypeClass(); 	  // check datatype
	    // std::cout << "type: " << type_class << std::endl;
	    if (type_class != H5T_FLOAT){
	    	std::cout <<"ERROR: Not float type in " << name << std::endl;
	    }		

	    dataset.read(odata, H5::PredType::NATIVE_FLOAT);
	    //std::cout << "done bias reading" << std::endl;

	 	//if (!odata){
		// 	std::cout << "ERROR: Failed to read bias data" << std::endl;
		// }
		file.close();


	}
	//catch failure caused by the H5File operations
	catch( H5::FileIException error )
	{
		std::cout << "in bias error" << std::endl;
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		std::cout << "in bias error" << std::endl;
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		std::cout << "in bias error" << std::endl;
		error.printError();
		return 1;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		std::cout << "in bias error" << std::endl;
		error.printError();
		return 1;
	}

	return 0;
}





