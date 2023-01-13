#include<bits/stdc++.h>
using namespace std;

bool is_number(const std::string& s){
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void convert_input(string filename){
    fstream database(filename);
    ofstream output("db_converted.txt");
    ofstream mapping("db_mapping.txt");
    ofstream label_mapping("db_label_mapping.txt");
    string line;
    map<string, int> d;
    bool readVertices = false;
    int vertexId = 0;
    int numVertices = 0;
    int numGraphs = 0;
    while(getline(database, line)){
        if(line.size() == 0){
            continue;
        }
        if(line[0] == '#'){
           output << "t # " << numGraphs << endl;
           mapping << line.substr(1) + "\n";
           numGraphs++;
           readVertices = false;
        }
        else if(is_number(line)){
            // cout << line << endl;
            readVertices = !readVertices;
            vertexId = 0;
        }
        else if(readVertices){
            if(!d.count(line)){
                d[line] = numVertices;
                numVertices++;
            }
            output << "v " << vertexId << " " << d[line] << endl;
            vertexId++;
        }
        else{
            output << "e " << line << endl;
        }
    }
    for(auto x: d){
        label_mapping << x.first << " " << x.second << endl;
    }
    output.close();
    mapping.close();
    label_mapping.close();
}

int main(int argc, char* argv[]){
    string filename = argv[1];
    convert_input(filename);
    float frequency = 0.3;
    string command = "./gSpan-64 -f db_converted.txt -s " + to_string(frequency) + " -o -i";
    cout << command << endl;
    system(command.c_str());

}