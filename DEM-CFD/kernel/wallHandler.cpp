#include "wallHandler.h"

// Example method inside triangleWall (or a handler class)
void wallHandler::outputTriangleWallVTU(const std::string& dir,
                                        const size_t      iFrame,
                                        const size_t      iStep,
                                        double            timeStep)
{
    // Create output directory if needed
    MKDIR(dir.c_str());

    // Build file name: triangleWall_0000.vtu
    std::ostringstream fname;
    fname << dir << "/triangleWall_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) {
        throw std::runtime_error("Cannot open " + fname.str());
    }

    out << std::fixed << std::setprecision(10);

    // -------------------------------------------------------------------------
    // Gather data on host
    // NOTE: Replace the getter names below with your actual implementations.
    // -------------------------------------------------------------------------

    // Global vertex positions (already transformed to world coordinates)
    const std::vector<double3> verts = triangleWalls_.getGlobalVerticesHost();
    const size_t numPoints = verts.size();

    // Global triangle connectivity (indices into verts)
    const std::vector<int> tri_i0 = triangleWalls_.triangles().getIndex0Host();
    const std::vector<int> tri_i1 = triangleWalls_.triangles().getIndex1Host();
    const std::vector<int> tri_i2 = triangleWalls_.triangles().getIndex2Host();
    const size_t numTris = tri_i0.size();

    // Wall id for each triangle (same length as tri_i0)
    const std::vector<int> triWallId = triangleWalls_.triangles().getWallIndexHost();

    // Map wall-level materialID_ to per-triangle material IDs
    std::vector<int> triMaterialId(numTris, 0);
    const std::vector<int> wallMaterial = triangleWalls_.getMaterialIDHost();
    if (!wallMaterial.empty() && !triWallId.empty()) {
        for (size_t t = 0; t < numTris; ++t) {
            const int w = triWallId[t];
            if (w >= 0 && static_cast<size_t>(w) < wallMaterial.size()) {
                triMaterialId[t] = wallMaterial[w];
            }
        }
    }

    // -------------------------------------------------------------------------
    // VTK XML header
    // -------------------------------------------------------------------------
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        << "  <UnstructuredGrid>\n";

    // -------------------------------------------------------------------------
    // FieldData: global meta info such as TIME and STEP
    // -------------------------------------------------------------------------
    out << "    <FieldData>\n"
        << "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << iStep * timeStep << " </DataArray>\n"
        << "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << iStep << " </DataArray>\n"
        << "    </FieldData>\n";

    // One piece containing all points and cells
    out << "    <Piece NumberOfPoints=\"" << numPoints
        << "\" NumberOfCells=\"" << numTris << "\">\n";

    // -------------------------------------------------------------------------
    // Points section
    // -------------------------------------------------------------------------
    out << "      <Points>\n"
        << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (size_t i = 0; i < numPoints; ++i) {
        out << ' ' << verts[i].x
            << ' ' << verts[i].y
            << ' ' << verts[i].z;
    }
    out << "\n        </DataArray>\n"
        << "      </Points>\n";

    // -------------------------------------------------------------------------
    // Cells section: triangle connectivity
    // connectivity: i0 i1 i2  i0 i1 i2  ...
    // offsets: 3, 6, 9, ..., 3*numTris
    // types: 5 for VTK_TRIANGLE
    // -------------------------------------------------------------------------
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t t = 0; t < numTris; ++t) {
        out << ' ' << tri_i0[t]
            << ' ' << tri_i1[t]
            << ' ' << tri_i2[t];
    }
    out << "\n        </DataArray>\n";

    // offsets
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t t = 1; t <= numTris; ++t) {
        out << ' ' << (3 * t);
    }
    out << "\n        </DataArray>\n";

    // types: VTK_TRIANGLE = 5
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t t = 0; t < numTris; ++t) {
        out << " 5";
    }
    out << "\n        </DataArray>\n"
        << "      </Cells>\n";

    // -------------------------------------------------------------------------
    // CellData: per-triangle attributes (wallId, materialID, etc.)
    // -------------------------------------------------------------------------
    out << "      <CellData Scalars=\"materialID\">\n";

    // wall id
    if (!triWallId.empty()) {
        out << "        <DataArray type=\"Int32\" Name=\"wallID\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << triWallId[t];
        }
        out << "\n        </DataArray>\n";
    }

    // material id
    out << "        <DataArray type=\"Int32\" Name=\"materialID\" format=\"ascii\">\n";
    for (size_t t = 0; t < numTris; ++t) {
        out << ' ' << triMaterialId[t];
    }
    out << "\n        </DataArray>\n";

    out << "      </CellData>\n";

    // -------------------------------------------------------------------------
    // (Optional) PointData: e.g., vertex normals, etc. can be added here
    // -------------------------------------------------------------------------
    // out << "      <PointData>\n";
    // ... your per-vertex arrays ...
    // out << "      </PointData>\n";

    out << "    </Piece>\n"
        << "  </UnstructuredGrid>\n"
        << "</VTKFile>\n";
}