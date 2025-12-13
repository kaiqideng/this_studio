#include "ballHandler.h"
#include "ballIntegration.h"
#include "myStruct/myUtility/myFileEdit.h"

void ballHandler::ballNeighborSearch(const size_t maxThreadsPerBlock)
{
    launchBallNeighborSearch(ballInteractions_, ballInteractionMap_, 
    balls_,
    spatialGrids_,
    maxThreadsPerBlock,
    stream_);
}

void ballHandler::ball1stHalfIntegration(const double3 g, const double t, const size_t maxThreadsPerBlock)
{
    launchBall1stHalfIntegration(balls_, g, t, maxThreadsPerBlock, stream_);
}

void ballHandler::ballContactCalculation(contactModelParameters &contactModelParams, const double t, const size_t maxThreadsPerBlock)
{
    launchBallContactCalculation(ballInteractions_, 
    bondedBallInteractions_, 
    balls_, 
    contactModelParams, 
    ballInteractionMap_,
    t, 
    maxThreadsPerBlock, 
    stream_);
}

void ballHandler::ball2ndHalfIntegration(const double3 g, const double t, const size_t maxThreadsPerBlock)
{
    launchBall2ndHalfIntegration(balls_, g, t, maxThreadsPerBlock, stream_);
}

void ballHandler::outputBallVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
{
    MKDIR(dir.c_str());
    std::ostringstream fname;
    fname << dir << "/solidParticle_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());
    out << std::fixed << std::setprecision(10);

    const size_t N = balls_.hostSize();
    const std::vector<double3> p = balls_.positionVector();
    const std::vector<double3> v = balls_.velocityVector();
    const std::vector<double3> a = balls_.angularVelocityVector();
    const std::vector<double> r = balls_.radiusVector();
    const std::vector<int> materialID = balls_.materialIDVector();
    const std::vector<int> clumpID = balls_.clumpIDVector();
    
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << iStep << " </DataArray>\n"
        "    </FieldData>\n";

    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    out << "      <PointData Scalars=\"radius\">\n";

    out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << r[i];
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"clumpID\" format=\"ascii\">\n";
	for (int i = 0; i < N; ++i) out << ' ' << materialID[i];
	out << "\n        </DataArray>\n";

	out << "        <DataArray type=\"Int32\" Name=\"clumpID\" format=\"ascii\">\n";
	for (int i = 0; i < N; ++i) out << ' ' << clumpID[i];
	out << "\n        </DataArray>\n";

    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , v     },
		{ "angularVelocity", a     }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < v.size(); ++i)
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        out << "\n        </DataArray>\n";
    }

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}