#include<bits/stdc++.h>
#include <chrono>
#include <csignal>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>
#include "WindowsTime.h"
#include "VFLib.h"
#include "Options.hpp"
#include "VF3SubState.hpp"
using namespace std::chrono;
using namespace std;
using namespace vflib;
typedef int32_t int32_t;
// typedef vflib::VF3SubState<int32_t, int32_t, int32_t, int32_t> state_t;

bool is_number(const std::string& s){
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

bool vf3_subgraph_isomorphism(vector<int32_t> &v1, vector<pair<pair<int32_t, int32_t>, int32_t>> &edges1, vector<int32_t> &v2, vector<pair<pair<int32_t, int32_t>, int32_t>> &edges2){
    ARGLoader<int32_t, int32_t>* pattloader = new vflib::FastStreamARGLoader<int32_t, int32_t>(v1, edges1, true);
    ARGLoader<int32_t, int32_t>* targloader = new vflib::FastStreamARGLoader<int32_t, int32_t>(v2, edges2, true);
    ARGraph<int32_t, int32_t> patt_graph(pattloader);
    ARGraph<int32_t, int32_t> targ_graph(targloader);
    MatchingEngine<state_t >* me = new vflib::MatchingEngine<state_t >(true);;
    FastCheck<int32_t, int32_t, int32_t, int32_t > check(&patt_graph, &targ_graph);
    vector<MatchingSolution> solutions;
    if(check.CheckSubgraphIsomorphism()){
        NodeClassifier<int32_t, int32_t> classifier(&targ_graph);
        NodeClassifier<int32_t, int32_t> classifier2(&patt_graph, classifier);
        vector<uint32_t> class_patt = classifier2.GetClasses();
        vector<uint32_t> class_targ = classifier.GetClasses();
        uint32_t classes_count = classifier.CountClasses();
        VF3NodeSorter<int32_t, int32_t, SubIsoNodeProbability<int32_t, int32_t> > sorter(&targ_graph);
        std::vector<int32_t> sorted = sorter.SortNodes(&patt_graph);
        state_t s0(&patt_graph, &targ_graph, class_patt.data(), class_targ.data(), classes_count, sorted.data());
        me->FindAllMatchings(s0);
        me->GetSolutions(solutions);
        // for(auto it = solutions.begin(); it != solutions.end(); it++)
        // {
        //  std::cout<< me->SolutionToString(*it) << std::endl;
        // }
        if(me->GetSolutionsCount() > 0){
            return true;
        }
    }
    return false;
}

vector<int> output(vector<vector<int>> &db_vertices, vector<vector<pair<pair<int, int>, int>>> &db_edges, map<int, bool> &candidate_graphs, vector<int> &query_vertices, vector<pair<pair<int, int>, int>> &query_edges){
    vector<int> ans;
    for(int i = 0; i < db_vertices.size(); i++){
        if(!candidate_graphs.count(i)){
            continue;
        }
        auto val = vf3_subgraph_isomorphism(query_vertices, query_edges, db_vertices[i], db_edges[i]);
        if(val){
            // cout << "Found: " << mapping[id] << endl;
            ans.push_back(i);
        }
    }
    return ans;
}

void read_index(vector<vector<int>> &index_vertices, vector<vector<pair<pair<int, int>, int>>> &index_edges, vector<set<int>> &containing_graphs){
    fstream database("db_converted.txt.fp");
    string line;
    vector<int> vertices;
    vector<pair<pair<int, int>, int>> edges;
    set<int> graphs;
    bool start = true;
    while(getline(database, line)){
        if(line.size() == 0){
            continue;
        }
        if(line[0] == 't'){
            if(!start){
                index_vertices.push_back(vertices);
                index_edges.push_back(edges);
                containing_graphs.push_back(graphs);
            }
            start = false;
            // cout << graphs.size() << endl;
            vertices.clear();
            edges.clear();     
            graphs.clear();   
        }
        else{
            stringstream ss(line);
            string temp;
            vector<string> splitted;
            while(ss >> temp){
                splitted.push_back(temp);
            }
            if(splitted[0] == "v"){
                vertices.push_back(stoi(splitted[2]));
            }
            else if(splitted[0] == "e"){
                edges.push_back({{stoi(splitted[1]), stoi(splitted[2])},stoi(splitted[3])});
            }
            else if(splitted[0] == "x"){
                for(int i = 1; i < splitted.size(); i++){
                    graphs.insert(stoi(splitted[i]));
                }
            }
        }
    }
    index_vertices.push_back(vertices);
    index_edges.push_back(edges);
    containing_graphs.push_back(graphs);
}

void read_mapping(vector<int> &mapping, map<string, int> &d){
    fstream database("db_mapping.txt");
    string line;
    while(getline(database, line)){
        if(line.size() == 0){
            continue;
        }
        mapping.push_back(stoi(line));
    }
    database.close();

    fstream label_mapping("db_label_mapping.txt");
    while(getline(label_mapping, line)){
        if(line.size() == 0){
            continue;
        }
        stringstream ss(line);
        string temp;
        vector<string> splitted;
        while(ss >> temp){
            splitted.push_back(temp);
        }
        d[splitted[0]] = stoi(splitted[1]);
    }
    label_mapping.close();
}

void generate_candidates(vector<vector<int>> &index_vertices, vector<vector<pair<pair<int, int>, int>>> &index_edges, vector<set<int>> &containing_graphs, set<int> &candidate_graphs, vector<int> &query_vertices, vector<pair<pair<int, int>, int>> &query_edges){
    for(int i = 0; i < index_vertices.size(); i++){
        auto val = vf3_subgraph_isomorphism(index_vertices[i], index_edges[i], query_vertices, query_edges);
        if(val){
            set<int> temp;
            set_intersection(candidate_graphs.begin(), candidate_graphs.end(), containing_graphs[i].begin(), containing_graphs[i].end(), std::inserter(temp, temp.begin()));
            // cout << temp.size() << endl;
            candidate_graphs = temp;
            // cout << candidate_graphs.size() << endl;
        }
    }
}

void read_graph(string filename, vector<vector<int>> &graph_vertices, vector<vector<pair<pair<int, int>, int>>> &graph_edges){
    vector<int> vertices;
    vector<pair<pair<int, int>, int>> edges;
    fstream query(filename);
    string line;
    bool start = true;
    while(getline(query, line)){
        if(line.size() == 0){
            continue;
        }
        if(line[0] == 't'){
            if(!start){
                graph_vertices.push_back(vertices);
                graph_edges.push_back(edges);
            }
            start = false;
            vertices.clear();
            edges.clear();      
        }
        else{
            stringstream ss(line);
            string temp;
            vector<string> splitted;
            while(ss >> temp){
                splitted.push_back(temp);
            }
            if(splitted[0] == "v"){
                vertices.push_back(stoi(splitted[2]));
            }
            else if(splitted[0] == "e"){
                edges.push_back({{stoi(splitted[1]), stoi(splitted[2])},stoi(splitted[3])});
            }
        } 
    }
    graph_vertices.push_back(vertices);
    graph_edges.push_back(edges);
}

void convert_input(string filename, map<string, int> &d){
    fstream database(filename);
    ofstream output("converted.txt");
    ofstream mapping("mapping.txt");
    string line;
    
    bool readVertices = false;
    int vertexId = 0;
    int numVertices = d.size();
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
                // cout << line << endl;
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
    output.close();
    mapping.close();
}

int main(int argc, char* argv[]){
    vector<vector<int>> db_vertices;
    vector<vector<pair<pair<int, int>, int>>> db_edges;
    vector<vector<int>> index_vertices;
    vector<vector<pair<pair<int, int>, int>>> index_edges;
    vector<set<int>> containing_graphs;
    vector<int> mapping;
    map<string, int> d;
    
    // Read Index and Database
    read_mapping(mapping, d);
    read_index(index_vertices, index_edges, containing_graphs);
    read_graph("db_converted.txt", db_vertices, db_edges);

    // Read Query
    string query_file;
    cout << "Enter path to query file: ";
    cin >> query_file;
    auto start = high_resolution_clock::now();
    convert_input(query_file, d);
    vector<vector<int>> query_vertices;
    vector<vector<pair<pair<int, int>, int>>> query_edges;
    read_graph("converted.txt", query_vertices, query_edges);

    // Write output
    ofstream result("output_CS5190471.txt");

    for(int i = 0; i < query_vertices.size(); i++){
        auto tstart = high_resolution_clock::now();
        set<int> candidate_graphs;
        for(int i = 0; i < mapping.size(); i++){
            candidate_graphs.insert(i);
        }
        generate_candidates(index_vertices, index_edges, containing_graphs, candidate_graphs, query_vertices[i], query_edges[i]);
        map<int, bool> m;
        for(auto x: candidate_graphs){
            m[x] = true;
        }
        // cout << candidate_graphs.size() << endl;
        vector<int> ans = output(db_vertices, db_edges, m, query_vertices[i], query_edges[i]);
        for(int i = 0; i < ans.size(); i++){
            result << mapping[ans[i]] << "\t";
        }
        result << "\n";
        auto tend = high_resolution_clock::now();
        cout << "Time taken (in milliseconds) for graph: " << i << " " << duration_cast<milliseconds>(tend - tstart).count() << endl;
    }
    result.close();
    auto end = high_resolution_clock::now();
    cout << "Time taken (in milliseconds) for query file: " << duration_cast<milliseconds>(end - start).count() << endl;

    // vector<int32_t> v1 = {1, 1, 1, 1};
    // vector<pair<pair<int32_t, int32_t>, int32_t>> edges1 = {{{0, 1}, 1}, {{1, 2}, 1}, {{2, 3}, 1}, {{3, 0}, 1}};
    // vector<int32_t> v2 = {1, 1, 1};
    // vector<pair<pair<int32_t, int32_t>, int32_t>> edges2 = {{{0, 1}, 2}, {{0, 2}, 1}};
    // bool x = vf3_subgraph_isomorphism(v2, edges2, v1, edges1);
    // cout << x << endl;
    return 0;
}