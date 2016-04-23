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


namespace
{
//	template <typename IntegerType>
//	IntegerType bitsToInt( IntegerType& result, const unsigned char* bits, bool little_endian = true )
//	  {
//	  result = 0;
//	  if (little_endian)
//		for (int n = sizeof( result ); n >= 0; n--)
//		  result = (result << 8) +bits[ n ];
//	  else
//		for (unsigned n = 0; n < sizeof( result ); n++)
//		  result = (result << 8) +bits[ n ];
//	  return result;
//	  }

	double* ReadMNISTData(string fileName, int length)
	{
		fileName = "Data/" + fileName;

		ifstream input( fileName, std::ios::binary );

		vector<char> buffer((
		            std::istreambuf_iterator<char>(input)),
		            (std::istreambuf_iterator<char>()));

		int c = 0;
		for(int i = 0; i < buffer.size(); ++i)
		{
			unsigned int value = (unsigned char) buffer[i];
			double d = value / 255.0;

//			unsigned short _byteswap_ushort(unsigned short value);

			cout << d << " ";
//			cout << ((int)buffer[i] == 0 ? 0 : 1);

			c++;

			if(c >= 28)
			{
				cout << "\n";
				c = 0;
			}
		}

		cout << buffer.size() << "\n";
//		buffer.
//		ofstream outfile (fileName, ofstream::binary);

//		return ;
//
//		ifstream inFile;
//			size_t size = 0; // here
//
//			inFile.open(fileName, ios::in|ios::binary|ios::ate );
//			char* oData = 0;
//
//			inFile.seekg(0, ios::end); // set the pointer to the end
//			size = inFile.tellg() ; // get the length of the file
//			cout << "Size of file: " << size;
//			inFile.seekg(0, ios::beg); // set the pointer to the beginning
//
//			oData = new char[ size+1 ]; //  for the '\0'
//			inFile.read( oData, size );
//			oData[size] = '\0' ; // set '\0'
//			cout << " oData size: " << strlen(oData) << "\n";
//
//			//print data
//			for ( size_t i = 0; i < strlen(oData); i++ )
//			{
//				cout << "oData["<<i<<"] " << oData[i];
//				cout << "\n";
//				cout << oData[i] << " + 'a' = " << ( oData[i] + 'a' );
//				cout << "\n\n";
//
//			}




//		cout << fileName << "\n";
//
////		ofstream outfile (fileName, ofstream::binary);
//
//		ifstream inFile;
//		size_t size = 0; // here
//
////		inFile.
//
//		inFile.open(fileName, ios::in|ios::binary|ios::ate );
//
//		cout << "Fail " << inFile.failbit << "\n";
//
////		char* oData = 0;
////
////		inFile.seekg(0, ios::end); // set the pointer to the end
////		size = inFile.tellg() ; // get the length of the file
////		cout << "Size of file: " << size << "\n";
////		inFile.seekg(0, ios::beg); // set the pointer to the beginning
//
////		cout << "File = " << inFile << "\n";
//		char* oData[length];
//
//		fread(oData, length, 1, fileName);
//
//
////		while(inFile.is_open()) // To get you all the lines.
////		{
////			inFile.seekg(0, ios::end);
////			size = inFile.tellg();
////			inFile.seekg(0, ios::beg);
////
////			oData = new char[size+1];
////			inFile.read(oData, size);
////			oData[size] = '\0';
////
//////			for(int i = 0; i < size; ++i)
//////			{
//////				cout << oData[i] << "\n";
//////			}
////		}
//		inFile.close();
//
//		cout << "begin trace\n";
//
//		for(int i = 0; i < size; ++i)
//		{
//			cout << oData[i] << "\n";
//		}
//
////		cout << oData << "\n";
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
