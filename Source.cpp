#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ctime>
#include "omp.h"

using namespace std;

// Функция ищет максимальный элемент в {col} столбце матрицы {matrix}.
int col_max(const vector<vector<double>>& matrix, int col) {
	int n = matrix.size();
	double max = abs(matrix[col][col]);
	int maxPos = col;

#pragma omp parallel shared(matrix) 
	{
		// локальный максимум в потоке.
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
#pragma omp critical	// ищем максимум среди всех потоков.
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
		// локальный максимум в потоке.
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
		//#pragma omp critical	// ищем максимум среди всех потоков.
		{
			if (max < loc_max) {
				max = loc_max;
				maxPos = loc_max_pos;
			}
		}
	}
	return maxPos;
}

// Функция меняет местами {origin} и {sub} строки матрицы.
void swap(vector<vector<double>>& matrix, int origin, int sub) {
	vector<double> temp = matrix[origin];
	matrix[origin] = matrix[sub];
	matrix[sub] = temp;
}

// Функция, вычитающая i-ю строку из остальных строк, находящихся ниже неё так, чтобы i-й элемент каждой строки был равен 0.
void elementaryСonversion(vector<vector<double>>& matrix, int mainRow) {
	int n = matrix.size();

	// параллельно вычитаем {mainRow} строку из всех строк, ниже неё.
#pragma omp parallel for
	for (int j = mainRow + 1; j < n; ++j) {
		// множитель для {mainRow} строки.
		double mul = -matrix[j][mainRow] / matrix[mainRow][mainRow];
		for (int k = mainRow; k < n; ++k)
			matrix[j][k] += matrix[mainRow][k] * mul;
	}

}

void elementaryСonversionSimple(vector<vector<double>>& matrix, int mainRow) {
	int n = matrix.size();

	// параллельно вычитаем {mainRow} строку из всех строк, ниже неё.
	//#pragma omp parallel for
	for (int j = mainRow + 1; j < n; ++j) {
		// множитель для {mainRow} строки.
		double mul = -matrix[j][mainRow] / matrix[mainRow][mainRow];
		for (int k = mainRow; k < n; ++k)
			matrix[j][k] += matrix[mainRow][k] * mul;
	}

}

// Функция, приводящая матрицу, переданную в неё к треугольному виду.
int triangulation(vector<vector<double>>& matrix) {
	int n = matrix.size();
	int swapCount = 0;

	// проходимся по всей матрице.
	for (int i = 0; i < n; ++i) {

		// находим строку, в которой i-й элемент самый большой в своем столбце.
		// ПАРАЛЛЕЛЬНО.
		int sub = col_max(matrix, i);


		// меняем найденную строку с i-й местами.
		if (sub != i) {
			swap(matrix, i, sub);
			swapCount++;
		}



		// вычитаем i-ю строку из остальных строк, находящихся ниже неё так чтобы i-й элемент каждой строки был равен 0.
		// ПАРАЛЛЕЛЬНО.
		elementaryСonversion(matrix, i);
	}
	return swapCount;
}

int triangulationSimple(vector<vector<double>>& matrix) {
	int n = matrix.size();
	int swapCount = 0;

	// проходимся по всей матрице.
	for (int i = 0; i < n; ++i) {

		// находим строку, в которой i-й элемент самый большой в своем столбце.
		// ПАРАЛЛЕЛЬНО.
		int sub = col_maxSimple(matrix, i);


		// меняем найденную строку с i-й местами.
		if (sub != i) {
			swap(matrix, i, sub);
			swapCount++;
		}


		// вычитаем i-ю строку из остальных строк, находящихся ниже неё так чтобы i-й элемент каждой строки был равен 0.
		// ПАРАЛЛЕЛЬНО.
		elementaryСonversionSimple(matrix, i);
	}
	return swapCount;
}

// Функция, вычисляющая определитель переданной в нее матрицы.
double calculateGaussDeterminant(vector<vector<double>> matrix) {
	int n = matrix.size();

	// приводим матрицу к треугольному виду.
	int swapCount = triangulation(matrix);

	// подсчитываем определитель.
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

	// приводим матрицу к треугольному виду.
	int swapCount = triangulationSimple(matrix);

	// подсчитываем определитель.
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


// Функция, читающая матрицу {matrix} из файла.
void readMatrixFromFile(vector<vector<double>>& matrix, ifstream& input) {
	// первая строка в файле - размерность матрицы.
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
		// запускаем многопоточную версию функции вычисления определителя методом Гаусса.
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

	// инициализируем матрицу с которой будем работать.
	vector<vector<double>> matrix(n, vector<double>(n));

	// считываем матрицу из файла с именем {argv[1]}.
	readMatrixFromFile(matrix, input);

	double determinant = 0;
		
	double time1 = checkTime(true, matrix, determinant);
	double time2 = checkTime(false, matrix, determinant);

	output << "При размере матрицы == " << n << endl;
	output << "Определитель == " << determinant << endl;
	output << "Время работы обычной программы : " << time2 << " cек." << endl;
	output << "Время работы многопоточной программы : " << time1 << " cек.";
}