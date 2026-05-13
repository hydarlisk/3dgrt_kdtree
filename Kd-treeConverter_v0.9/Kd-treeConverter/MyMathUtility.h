/**************************************************************
  File name: MyMathUtility.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#define TO_DEGREE 57.29577951308232
#define TO_RADIAN  0.01745329251994330

#define MyABS(x)     (((x) < 0)? -(x) : (x))
#define MyMAX(a,b)   (((a) > (b)) ? (a) : (b))
#define MyMIN(a,b)   (((a) < (b)) ? (a) : (b))

#include <iostream>

class MyVec3 {
public:
    union {
        struct { float x, y, z; };
        float data[3];
    };

    MyVec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    float& operator[](int index) {
        if (index < 0 || index >= 3) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    const float& operator[](int index) const {
        if (index < 0 || index >= 3) throw std::out_of_range("Index out of bounds");
        return data[index];
    }
    MyVec3 operator+(const MyVec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    MyVec3 operator-(const MyVec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
    MyVec3 operator*(float s) const { return { x * s, y * s, z * s }; }

    float dot(const MyVec3& v) const { return x * v.x + y * v.y + z * v.z; }

    MyVec3 cross(const MyVec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    void print() const {
        std::cout << "[" << x << ", " << y << ", " << z << "]" << std::endl;
    }
};

class MyMat33 {
public:
    MyVec3 rows[3];

    MyMat33() {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rows[i][j] = 0;
    }

    static MyMat33 identity() {
        MyMat33 res;
        res.rows[0][0] = res.rows[1][1] = res.rows[2][2] = 1.0f;
        return res;
    }


    /*** operators ***/
    MyVec3& operator[](int index) {
        if (index < 0 || index >= 3) throw std::out_of_range("Index out of bounds");
        return rows[index];
    }
    MyMat33 operator*(const float& other) const {
        MyMat33 res{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res.rows[i][j] = rows[i][j] * other;
            }
        }
        return res;
    }
    // mat33 * mat33
    MyMat33 operator*(const MyMat33& other) const {
        MyMat33 res{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    res.rows[i][j] += rows[i][k] * other.rows[k][j];
                }
            }
        }
        return res;
    }
    // vec3 * mat33
    friend MyVec3 operator*(const MyVec3& v, const MyMat33& m) {
        MyVec3 res(0, 0, 0);
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                res[j] += v[i] * m.rows[i][j];
            }
        }
        return res;
    }
    // mat33 * vec3
    MyVec3 operator*(const MyVec3& v) const {
        MyVec3 res;
        for (int i = 0; i < 3; i++) {
            res[i] = rows[i][0] * v.x + rows[i][1] * v.y + rows[i][2] * v.z;
        }
        return res;
    }
    // mat33 * mat33T
    MyMat33 multTranspose() const {
        MyMat33 res;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res[i][j] = rows[i][0] * rows[j][0] +
                    rows[i][1] * rows[j][1] +
                    rows[i][2] * rows[j][2];
            }
        }
        return res;
    }

    MyMat33 transpose() const {
        MyMat33 res;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res[j][i] = rows[i][j];
            }
        }
        return res;
    }

    void print() const {
        for (int i = 0; i < 3; i++) {
            std::cout << "| " << rows[i][0] << " " << rows[i][1] << " " << rows[i][2] << " |" << std::endl;
        }
    }
};


void dMyVecCrossProduct(double *, double *, double *);
void dMyVecNormalize(double *); 
double dMyVecLength(double *v);
double dMyVecDotProduct(double *, double *); 

void fMyVecCrossProduct(float *, float *, float *);
void fMyVecNormalize(float *); 
float fMyVecLength(float *);
float fMyVecDotProduct(float *, float *);  
void fMyVecElaborateFaceNormal(float *, float *, float *, float *);

