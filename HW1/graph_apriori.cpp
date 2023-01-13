#include<bits/stdc++.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

void generateCandidates(map<vector<int>, int> &frequent_sets, int k, int n, int support,vector<vector<int>> &candidates){
	for(auto it1 = frequent_sets.begin(); it1 != frequent_sets.end(); it1++){
		for(auto it2 = it1; it2 != frequent_sets.end(); it2++){
			vector<int> v1 = it1->first;
			vector<int> v2 = it2->first;

			bool are_mergable = true;
			for(int i = 0; i < k - 2; i++){
				if(v1[i] != v2[i]){
					are_mergable = false;
				}
			}

			if(are_mergable && v1[k - 2] == v2[k - 2]){
				are_mergable = false;
			}

			if(are_mergable){
				vector<int> candidate;
				for(int i = 0; i < k - 2; i++){
					candidate.push_back(v1[i]);
				}
				if(v1[k - 2] < v2[k - 2]){
					candidate.push_back(v1[k - 2]);
					candidate.push_back(v2[k - 2]);
				}
				else{
					candidate.push_back(v2[k - 2]);
					candidate.push_back(v1[k - 2]);
				}

				bool is_candidate = true;
				for(int i = 0; i < k - 1; i++){
					int removed = candidate[i];
					candidate.erase(candidate.begin() + i);
					if(frequent_sets[candidate]*100 < support*n){
						is_candidate = false;
						break;
					}
					candidate.insert(candidate.begin() + i, removed);
				}
				if(is_candidate){
					candidates.push_back(candidate);
				}
			}
		}	
	}
	return;
}

void apriori(string file, int support,vector<vector<int>> &ans){

	// Stores frequent item sets of size 1
	set<vector<int>>unit_size_candidates;
	map<vector<int>, int> frequent_sets;

	int num_transactions = 0;

	// Generates candidates of size 1
	fstream inp(file);
	string s;
	while (getline (inp, s)){
		stringstream ss(s);
		string temp;
		num_transactions++;
		while(ss >> temp){
			vector<int> item;
			item.push_back(stoi(temp));
			unit_size_candidates.insert(item);
			frequent_sets[item]++;
		}
	}
	inp.close();

	int distinct = frequent_sets.size();

	// Check frequency of candidate item sets
	for(auto candidate: unit_size_candidates){
		if (frequent_sets[candidate]*100 < num_transactions*support){
			frequent_sets.erase(candidate);
		}
		else{
			ans.push_back(candidate);
		}
	}

	// For item set of size k
	int k = 2;
	while(true){
		vector<vector<int>> candidates;
		if(k == distinct + 1){
			break;
		}
		generateCandidates(frequent_sets, k, num_transactions, support, candidates);
		if(candidates.size() == 0){
			break;
		}
		frequent_sets.clear();

		// For each transaction
		fstream inp(file);
		string s;
		vector<int> count(candidates.size());
		while (getline (inp, s)){
			stringstream ss(s);
			string temp;
			vector<int> transaction;
			while(ss >> temp){
				transaction.push_back(stoi(temp));
			}

			sort(transaction.begin(), transaction.end());
        	vector<int> comp(k);
        	
        	int i = 0;
			// For each candidate
			for(auto candidate: candidates){
            	if(includes(transaction.begin(), transaction.end(), candidate.begin(), candidate.end())){
                	count[i]++;
            	}
            	i++;
			}
		}
		inp.close();

		// Check frequency of each itemset
		int i = 0;
		for(auto candidate: candidates){
			if (count[i]*100 >= num_transactions*support){
				ans.push_back(candidate);
				frequent_sets[candidate] = count[i];
			}
			i++;
		}
		k++;
	}
	return;
}

int main(int argc, char* argv[]){
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);

	vector<vector<int>> ans;
	vector<double> times;
	vector<int> thresholds = {90, 50, 25, 10, 5};

	string dataset = argv[1];
	string outName = "apriori.txt";

	for(int i = 0; i < 5; i++){
		auto start = high_resolution_clock::now();
		apriori(dataset, thresholds[i], ans);
		auto end = high_resolution_clock::now();
		times.push_back(duration_cast<microseconds>(end - start).count());
	}

	ofstream output;
	output.open(outName);
	for(int i = 0; i < 5; i++){
		output << times[i]*0.001 << " ";
	}
	output << "\n";
	output.close();
	
}