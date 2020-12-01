#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ctime>
#include "omp.h"

using namespace std;

// ������� ���� ������������ ������� � {col} ������� ������� {matrix}.
int col_max(const vector<vector<double>>& matrix, int col) {
	int n = matrix.size();
	double max = abs(matrix[col][col]);
	int maxPos = col;

#pragma omp parallel shared(matrix) 
	{
		// ��������� �������� � ������.
		double loc_max = max;
		int loc_max_pos = maxPos;
#pragma omp for
		for (int i = col + 1; i < n; ++i) {
			double element = abs(matrix[i][col]);
			if (element > loc_max) {
				loc_max = element;
				loc_max_pos = i;
			}
		}
#pragma omp critical	// ���� �������� ����� ���� �������.
		{
			if (max < loc_max) {
				max = loc_max;
				maxPos = loc_max_pos;
			}
		}
	}
	return maxPos;
}

int col_maxSimple(const vector<vector<double>>& matrix, int col) {
	int n = matrix.size();
	double max = abs(matrix[col][col]);
	int maxPos = col;

	//#pragma omp parallel shared(matrix) 
	{
		// ��������� �������� � ������.
		double loc_max = max;
		int loc_max_pos = maxPos;
		//#pragma omp for
		for (int i = col + 1; i < n; ++i) {
			double element = abs(matrix[i][col]);
			if (element > loc_max) {
				loc_max = element;
				loc_max_pos = i;
			}
		}
		//#pragma omp critical	// ���� �������� ����� ���� �������.
		{
			if (max < loc_max) {
				max = loc_max;
				maxPos = loc_max_pos;
			}
		}
	}
	return maxPos;
}

// ������� ������ ������� {origin} � {sub} ������ �������.
void swap(vector<vector<double>>& matrix, int origin, int sub) {
	vector<double> temp = matrix[origin];
	matrix[origin] = matrix[sub];
	matrix[sub] = temp;
}

// �������, ���������� i-� ������ �� ��������� �����, ����������� ���� �� ���, ����� i-� ������� ������ ������ ��� ����� 0.
void elementary�onversion(vector<vector<double>>& matrix, int mainRow) {
	int n = matrix.size();

	// ����������� �������� {mainRow} ������ �� ���� �����, ���� ��.
#pragma omp parallel for
	for (int j = mainRow + 1; j < n; ++j) {
		// ��������� ��� {mainRow} ������.
		double mul = -matrix[j][mainRow] / matrix[mainRow][mainRow];
		for (int k = mainRow; k < n; ++k)
			matrix[j][k] += matrix[mainRow][k] * mul;
	}

}

void elementary�onversionSimple(vector<vector<double>>& matrix, int mainRow) {
	int n = matrix.size();

	// ����������� �������� {mainRow} ������ �� ���� �����, ���� ��.
	//#pragma omp parallel for
	for (int j = mainRow + 1; j < n; ++j) {
		// ��������� ��� {mainRow} ������.
		double mul = -matrix[j][mainRow] / matrix[mainRow][mainRow];
		for (int k = mainRow; k < n; ++k)
			matrix[j][k] += matrix[mainRow][k] * mul;
	}

}

// �������, ���������� �������, ���������� � �� � ������������ ����.
int triangulation(vector<vector<double>>& matrix) {
	int n = matrix.size();
	int swapCount = 0;

	// ���������� �� ���� �������.
	for (int i = 0; i < n; ++i) {

		// ������� ������, � ������� i-� ������� ����� ������� � ����� �������.
		// �����������.
		int sub = col_max(matrix, i);


		// ������ ��������� ������ � i-� �������.
		if (sub != i) {
			swap(matrix, i, sub);
			swapCount++;
		}



		// �������� i-� ������ �� ��������� �����, ����������� ���� �� ��� ����� i-� ������� ������ ������ ��� ����� 0.
		// �����������.
		elementary�onversion(matrix, i);
	}
	return swapCount;
}

int triangulationSimple(vector<vector<double>>& matrix) {
	int n = matrix.size();
	int swapCount = 0;

	// ���������� �� ���� �������.
	for (int i = 0; i < n; ++i) {

		// ������� ������, � ������� i-� ������� ����� ������� � ����� �������.
		// �����������.
		int sub = col_maxSimple(matrix, i);


		// ������ ��������� ������ � i-� �������.
		if (sub != i) {
			swap(matrix, i, sub);
			swapCount++;
		}


		// �������� i-� ������ �� ��������� �����, ����������� ���� �� ��� ����� i-� ������� ������ ������ ��� ����� 0.
		// �����������.
		elementary�onversionSimple(matrix, i);
	}
	return swapCount;
}

// �������, ����������� ������������ ���������� � ��� �������.
double calculateGaussDeterminant(vector<vector<double>> matrix) {
	int n = matrix.size();

	// �������� ������� � ������������ ����.
	int swapCount = triangulation(matrix);

	// ������������ ������������.
	double det = 1;
	if (swapCount % 2 == 1)
		det = -1;
#pragma omp parallel reduction (*: det)
	{
# pragma omp for
		for (int i = 0; i < n; ++i) {
			det *= matrix[i][i];
		}


	}
	return round(det);
}

double calculateGaussDeterminantSimple(vector<vector<double>> matrix) {
	int n = matrix.size();

	// �������� ������� � ������������ ����.
	int swapCount = triangulationSimple(matrix);

	// ������������ ������������.
	double det = 1;
	if (swapCount % 2 == 1)
		det = -1;
	//#pragma omp parallel reduction (*: det)
	{
		//# pragma omp for
		for (int i = 0; i < n; ++i) {
			det *= matrix[i][i];
		}


	}
	return round(det);
}


// �������, �������� ������� {matrix} �� �����.
void readMatrixFromFile(vector<vector<double>>& matrix, ifstream& input) {
	// ������ ������ � ����� - ����������� �������.
	int n = matrix[0].size();

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			input >> matrix[i][j];
		}
	}
}

double checkTime(bool isMP, vector<vector<double>> matrix, double& determinant) {
	double start = clock();

	if (isMP) {
		// ��������� ������������� ������ ������� ���������� ������������ ������� ������.
		determinant = calculateGaussDeterminant(matrix);
	}
	else {
		determinant = calculateGaussDeterminantSimple(matrix);
	}
	double end = clock();

	return (double)(end - start) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[]) {

	ifstream input;
	ofstream output;
	input.open("test_100.txt");
	output.open(argv[2]);

	int n;
	input >> n;

	// �������������� ������� � ������� ����� ��������.
	vector<vector<double>> matrix(n, vector<double>(n));

	// ��������� ������� �� ����� � ������ {argv[1]}.
	readMatrixFromFile(matrix, input);

	double determinant = 0;
		
	double time1 = checkTime(true, matrix, determinant);
	double time2 = checkTime(false, matrix, determinant);

	output << "��� ������� ������� == " << n << endl;
	output << "������������ == " << determinant << endl;
	output << "����� ������ ������� ��������� : " << time2 << " c��." << endl;
	output << "����� ������ ������������� ��������� : " << time1 << " c��.";
}