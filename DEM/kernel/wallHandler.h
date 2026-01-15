#pragma once
#include "cudaKernel/myStruct/wall.h"
#include "cudaKernel/myStruct/spatialGrid.h"
#include "cudaKernel/myStruct/myUtility/myFileEdit.h"
#include "cudaKernel/ballMeshWallIntegration.h"

class wallHandler
{
public:
    wallHandler()
    {
        uploadFlag_ = false;
    }

    ~wallHandler() = default;

    void addTriangleMeshWall(const std::vector<double3> &vertices, 
    const std::vector<int3> &triIndices, 
    const double3 &posistion, 
    const double3 &velocity, 
    const double3 &angularVelocity, 
    int matirialID,
    cudaStream_t stream)
    {
        if(!uploadFlag_)
        {
            meshWalls_.download(stream);
            uploadFlag_ = true;
        }
        meshWalls_.addWallFromMesh(vertices, 
        triIndices, 
        posistion, 
        velocity, 
        angularVelocity, 
        make_quaternion(1, 0, 0, 0), 
        matirialID);
    }

    meshWall& getMeshWalls() {return meshWalls_;}

    void upload(cudaStream_t stream)
    {
        if(uploadFlag_)
        {
            meshWalls_.upload(stream);
            uploadFlag_ = false;
        }
    }

    void outputMeshWallVTU(const std::string& dir,
    const size_t iFrame,
    const size_t iStep,
    double time)
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/triangleWall_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        std::ofstream out(fname.str().c_str());
        if (!out) {
            throw std::runtime_error("Cannot open " + fname.str());
        }

        out << std::fixed << std::setprecision(10);

        std::vector<double3> verts = meshWalls_.globalVerticesVector();
        const size_t numPoints = verts.size();

        const std::vector<int> tri_i0 = meshWalls_.triangles().index0Vector();
        const std::vector<int> tri_i1 = meshWalls_.triangles().index1Vector();
        const std::vector<int> tri_i2 = meshWalls_.triangles().index2Vector();
        const size_t numTris = tri_i0.size();

        const std::vector<int> triWallId = meshWalls_.triangles().wallIndexVector();

        std::vector<int> triMaterialId(numTris, 0);
        const std::vector<int> wallMaterial = meshWalls_.materialIDVector();
        if (!wallMaterial.empty() && !triWallId.empty()) {
            for (size_t t = 0; t < numTris; ++t) {
                const int w = triWallId[t];
                if (w >= 0 && static_cast<size_t>(w) < wallMaterial.size()) {
                    triMaterialId[t] = wallMaterial[w];
                }
            }
        }

        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            << "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            << "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            << "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            << "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << numPoints
            << "\" NumberOfCells=\"" << numTris << "\">\n";

        out << "      <Points>\n"
            << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < numPoints; ++i) {
            out << ' ' << verts[i].x
                << ' ' << verts[i].y
                << ' ' << verts[i].z;
        }
        out << "\n        </DataArray>\n"
            << "      </Points>\n";

        out << "      <Cells>\n";

        out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << tri_i0[t]
                << ' ' << tri_i1[t]
                << ' ' << tri_i2[t];
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t t = 1; t <= numTris; ++t) {
            out << ' ' << (3 * t);
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << " 5";
        }
        out << "\n        </DataArray>\n"
            << "      </Cells>\n";

        out << "      <CellData Scalars=\"materialID\">\n";

        if (!triWallId.empty()) {
            out << "        <DataArray type=\"Int32\" Name=\"wallID\" format=\"ascii\">\n";
            for (size_t t = 0; t < numTris; ++t) {
                out << ' ' << triWallId[t];
            }
            out << "\n        </DataArray>\n";
        }

        out << "        <DataArray type=\"Int32\" Name=\"materialID\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << triMaterialId[t];
        }
        out << "\n        </DataArray>\n";

        out << "      </CellData>\n";

        out << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "</VTKFile>\n";
    }

    void integration(const double dt, const size_t maxThreads, cudaStream_t stream)
	{
        launchMeshWallIntegration(meshWalls_, 
        dt, 
        maxThreads, 
        stream);
	}

private:
    bool uploadFlag_ ;

    meshWall meshWalls_;
    spatialGrid spatialGrids_;
};