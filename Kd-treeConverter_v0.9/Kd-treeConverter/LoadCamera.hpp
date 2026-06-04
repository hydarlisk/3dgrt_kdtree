#pragma once

#include "OpenGLStuffs.h"

#include "../external/json.hpp"

#include <string>
#include <fstream>

using namespace std;
using nlohmann::json;

json loadJsonFromFile(const std::string& jsonPath) {
    std::ifstream fin(jsonPath);
    if (!fin.is_open()) {
        printf("Failed to open file : %s", jsonPath.c_str());
        exit(-1);
    }
    json j;
    fin >> j;
    return j;
}


void loadCameraJson(vector<Camera>& cameras, string jsonPath, Camera& camera) {
	json j = loadJsonFromFile(jsonPath);

    camera.fovy = 39.6;
    camera.near_c = 0.005;
    camera.far_c = 20.0;

    int i = 0;
    for (const auto& frame : j["frames"]) {
        const auto& matrix = frame["transform_matrix"];
        Camera cam;
        cam.uaxis[0] = matrix[0][0];
        cam.uaxis[1] = matrix[1][0];
        cam.uaxis[2] = matrix[2][0];
        cam.vaxis[0] = matrix[0][1];
        cam.vaxis[1] = matrix[1][1];
        cam.vaxis[2] = matrix[2][1];
        cam.naxis[0] = matrix[0][2];
        cam.naxis[1] = matrix[1][2];
        cam.naxis[2] = matrix[2][2];

        cam.pos[0] = matrix[0][3];
        cam.pos[1] = matrix[1][3];
        cam.pos[2] = matrix[2][3];

        cam.near_c = camera.near_c;
        cam.far_c = camera.far_c;
        cam.fovy = camera.fovy;
        cam.aspect = camera.aspect;

        cameras.push_back(cam);
    }
}