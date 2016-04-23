#ifndef READMNIST_H_
#define READMNIST_H_
#include <stdio.h>
#include <iostream>
#include <array>
using namespace std;
#include <vector>
#include <fstream>
#include <cstring> // for std::strlen
#include <cstddef>
#include "Globals.h"
#include "SharedData.cu"

namespace
{

class MNISTElement
{
	public:
	MNISTElement()
	{
		Data = new SharedData<double>(MNIST_ELEMENT_SIZE);
	}

	void Init()
	{
		Data->CopyToDevice();
	}

	SharedData<double>* Data;

	virtual ~MNISTElement()
	{

	}
};

	void ReadMNISTData(string fileName, array<MNISTElement*, MNIST_ELEMENTS_TO_LOAD> output)
	{
		fileName = "Data/" + fileName;

		cout << "ReadMNISTData\n";

		ifstream input( fileName, std::ios::binary );

		vector<char> buffer((
		            std::istreambuf_iterator<char>(input)),
		            (std::istreambuf_iterator<char>()));

//		const int elementLength = 28 * 28;

//		array<double*, MNIST_ELEMENTS_TO_LOAD> values;

		int b = 0;
		int totalLength = buffer.size();
		for(int i = 0; i < MNIST_ELEMENTS_TO_LOAD; ++i)
		{
//			HostData = (T *) calloc(1, TotalBytes);
//			values[i] = (double*) calloc(MNIST_ELEMENT_SIZE, sizeof(double));

			for(int x = 0; x < MNIST_ELEMENT_SIZE; ++x)
			{
//				if(b < totalLength)
//				{
				unsigned int value = (unsigned char) buffer[b];
				double d = value / 255.0;

//				cout << d << "\n";

				output[i]->Data->HostData[x] = d;

				b++;
//				}
			}
		}

		cout << "Finished load \n";
		cout << MNIST_ELEMENTS_TO_LOAD << " " << output.size() << "\n";

#if PRINT_MNIST_DATA
		for(int i = 0; i < MNIST_ELEMENTS_TO_LOAD; ++i)
		{
			int c = 0;
			for(int x = 0; x < MNIST_ELEMENT_SIZE; ++x)
			{
				cout << output[i]->Data->HostData[x] << " ";

				c++;

				if(c >= 28)
				{
					cout << "\n";
					c = 0;
				}
			}
			cout << "\n";
		}

		cout << buffer.size() << "\n";
#endif


//		return values;
	}
}

//unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
//    auto reverseInt = [](int i) {
//        unsigned char c1, c2, c3, c4;
//        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
//        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
//    };
//
//    typedef unsigned char uchar;
//
//    ifstream file(full_path);
//
//    if(file.is_open()) {
//        int magic_number = 0, n_rows = 0, n_cols = 0;
//
//        file.read((char *)&magic_number, sizeof(magic_number));
//        magic_number = reverseInt(magic_number);
//
//        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");
//
//        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
//        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
//        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
//
//        image_size = n_rows * n_cols;
//
//        uchar** _dataset = new uchar*[number_of_images];
//        for(int i = 0; i < number_of_images; i++) {
//            _dataset[i] = new uchar[image_size];
//            file.read((char *)_dataset[i], image_size);
//        }
//        return _dataset;
//    } else {
//        throw runtime_error("Cannot open file `" + full_path + "`!");
//    }
//}
//
//uchar** read_mnist_labels(string full_path, int& number_of_labels) {
//    auto reverseInt = [](int i) {
//        unsigned char c1, c2, c3, c4;
//        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
//        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
//    };
//
//    typedef unsigned char uchar;
//
//    ifstream file(full_path);
//
//    if(file.is_open()) {
//        int magic_number = 0;
//        file.read((char *)&magic_number, sizeof(magic_number));
//        magic_number = reverseInt(magic_number);
//
//        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");
//
//        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
//
//        uchar* _dataset = new uchar[number_of_labels];
//        for(int i = 0; i < number_of_labels; i++) {
//            file.read((char*)&_dataset[i], 1);
//        }
//        return _dataset;
//    } else {
//        throw runtime_error("Cannot open file `" + full_path + "`!");
//    }
//}


//class ReadMNIST
//{
//public://	int ReverseInt (int i)
////	{
////		unsigned char ch1, ch2, ch3, ch4;
////		ch1=i&255;
////		ch2=(i>>8)&255;
////		ch3=(i>>16)&255;
////		ch4=(i>>24)&255;
////		return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
////	}
////
////	void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector< vector<double> > &arr)
////	{
////		arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
////		ifstream file ("C:\\Users\admin\Documents\t10k-images.idx3-ubyte", ios::binary);
////		if (file.is_open())
////		{
////			int magic_number=0;
////			int number_of_images=0;
////			int n_rows=0;
////			int n_cols=0;
////			file.read((char*)&magic_number,sizeof(magic_number));
////			magic_number= ReverseInt(magic_number);
////			file.read((char*)&number_of_images,sizeof(number_of_images));
////			number_of_images= ReverseInt(number_of_images);
////			file.read((char*)&n_rows,sizeof(n_rows));
////			n_rows= ReverseInt(n_rows);
////			file.read((char*)&n_cols,sizeof(n_cols));
////			n_cols= ReverseInt(n_cols);
////			for(int i=0;i<number_of_images;++i)
////			{
////				for(int r=0;r<n_rows;++r)
////				{
////					for(int c=0;c<n_cols;++c)
////					{
////						unsigned char temp=0;
////						file.read((char*)&temp,sizeof(temp));
////						arr[i][(n_rows*r)+c]= (double)temp;
////					}
////				}
////			}
////		}
////	}
//}

#endif
