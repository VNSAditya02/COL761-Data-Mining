#include<bits/stdc++.h>
using namespace std;

struct Node{
	Node* parent;
	vector<Node*> children;
	int data = 0;
	int frequency = 0;
	Node* next;

	Node(int data, int frequency, Node* parent){
		this->data = data;
		this->frequency = frequency;
		this->parent = parent;
	}
};

void modifyTree(Node* root, vector<pair<int, int>> &transaction, int pos, map<int, Node*> &headerTable){
	if(pos == transaction.size()){
		return;
	}
	for(int i = 0; i < root->children.size(); i++){
		if(root->children[i]->data == transaction[pos].first){
			root->children[i]->frequency += transaction[pos].second;
			modifyTree(root->children[i], transaction, pos + 1, headerTable);
			return;
		}
	}
	
	for(int i = pos; i < transaction.size(); i++){
		Node* temp = new Node(transaction[i].first, transaction[i].second, root);
		root->children.push_back(temp);
		root = temp;
		temp->next = headerTable[transaction[i].first];
		headerTable[transaction[i].first] = temp;
	}
	return;
}

void bfs(Node* root){
	queue<Node*> q;
	q.push(root);
	while(!q.empty()){
		Node* temp = q.front();
		q.pop();
		cout << temp->data << ": " << temp->frequency << endl;
		for(int i = 0; i < temp->children.size(); i++){
			q.push(temp->children[i]);
		}
	}
}

void printHeaderTable(map<int, Node*> &headerTable){
	for(auto x: headerTable){
		Node* temp = x.second;
		while(temp != NULL){
			cout << temp->data << " " << temp->frequency << "; ";
			temp = temp->next;
		}
		cout << endl;
	}
}

Node* createTree(string file, map<int, int> &count, map<int, Node*> &headerTable){
	Node* root = new Node(-1, 0, NULL);

	// Read each transaction
	fstream inp2(file);
	string s;
	while (getline (inp2, s)){
		stringstream ss(s);
		string temp;
		vector<pair<int, int>> v;
		while(ss >> temp){
			if(count.count(stoi(temp))){
				v.push_back({-1*count[stoi(temp)], stoi(temp)});
			}
		}
		sort(v.begin(), v.end());
		vector<pair<int, int>> transaction;
		for(int i = 0; i < v.size(); i++){
			transaction.push_back({v[i].second, 1});
		}
		modifyTree(root, transaction, 0, headerTable);
	}
	inp2.close();
	// bfs(root);
	// printHeaderTable(headerTable);
	return root;
}

Node* createCondTree(vector<vector<pair<int, int>>> &paths, map<int, int> &count, map<int, Node*> &headerTable){
	Node* root = new Node(-1, 0, NULL);

	// Read each transaction
	for(int i = 0; i < paths.size(); i++){
		vector<pair<int, pair<int, int>>> v;
		for(int j = 0; j < paths[i].size(); j++){
			if(count.count(paths[i][j].first)){
				v.push_back({-1*count[paths[i][j].first], paths[i][j]});
			}
		}
		sort(v.begin(), v.end());
		vector<pair<int, int>> transaction;
		for(int i = 0; i < v.size(); i++){
			transaction.push_back({v[i].second.first, v[i].second.second});
		}
		modifyTree(root, transaction, 0, headerTable);
	}
	return root;
}

vector<pair<int, int>> getPath(Node* node, int freq){
	vector<pair<int, int>> path;
	node = node->parent;
	while(node->parent != NULL){
		path.push_back({node->data, freq});
		node = node->parent;
	}
	reverse(path.begin(), path.end());
	return path;
}

void mineTree(int data, int support, int num_transactions, set<int> condSet, map<int, Node*> &headerTable, vector<set<int>> &ans){
	vector<vector<pair<int, int>>> paths;
	Node* node = headerTable[data];
	while(node != NULL){
		paths.push_back(getPath(node, node->frequency));
		node = node->next;
	}

	map<int, int> condCount;
	map<int, Node*> condHeaderTable;
	set<int> distinctItems;
	vector<pair<int, int>> v;
	for(int i = 0; i < paths.size(); i++){
		for(int j = 0; j < paths[i].size(); j++){
			condCount[paths[i][j].first] += paths[i][j].second;
			distinctItems.insert(paths[i][j].first);
		}
	}

	for(auto itr = distinctItems.begin(); itr != distinctItems.end(); itr++){
		if(condCount[*itr]*100 < num_transactions*support){
			condCount.erase(*itr);
		}
		else{
			v.push_back({condCount[*itr], *itr});
			condSet.insert(*itr);
			ans.push_back(condSet);
			condSet.erase(*itr);
		}
	}

	if(v.size() == 0){
		return;
	}

	createCondTree(paths, condCount, condHeaderTable);
	sort(v.begin(), v.end());
	for(int i = 0; i < v.size(); i++){
		condSet.insert(v[i].second);
		mineTree(v[i].second, support, num_transactions, condSet, condHeaderTable, ans);
		condSet.erase(v[i].second);
	}
}

void fptree(string file, int support, vector<set<int>> &ans){
	int num_transactions = 0;

	// Store frequency of item sets
	map<int, int> count;
	set<int> distinctItems;
	map<int, Node*> headerTable;
	vector<pair<int, int>> v;
	fstream inp1(file);
	string s;
	while (getline (inp1, s)){
		stringstream ss(s);
		string temp;
		num_transactions++;
		while(ss >> temp){
			count[stoi(temp)]++;
			headerTable[stoi(temp)] = NULL;
			distinctItems.insert(stoi(temp));
		}
	}
	inp1.close();

	for(auto itr = distinctItems.begin(); itr != distinctItems.end(); itr++){
		if(count[*itr]*100 < num_transactions*support){
			count.erase(*itr);
		}
		else{
			set<int> S;
			S.insert(*itr);	
			ans.push_back(S);
			v.push_back({count[*itr], *itr});
		}
	}
	sort(v.begin(), v.end());
	createTree(file, count, headerTable);

	for(int i = 0; i < v.size(); i++){
		set<int> S;
		S.insert(v[i].second);
		mineTree(v[i].second, support, num_transactions, S, headerTable, ans);
	}
	
	
}

vector<string> customSort(vector<set<int>> &V){
	vector<string> ans;
	for(auto S: V){
		vector<string> temp;
		for(auto ele: S){
			temp.push_back(to_string(ele));
		}
		sort(temp.begin(), temp.end());
		string s;
		for(int i = 0; i < temp.size(); i++){
			s += temp[i] + " ";
		}
		ans.push_back(s);
	}	
	sort(ans.begin(), ans.end());
	return ans;
}

int main(int argc, char* argv[]){
	string dataset = argv[1];
	int support = stoi(argv[2]);
	string outName = argv[3];

	vector<set<int>> V;
	fptree(dataset, support, V);
	vector<string> ans = customSort(V);

	ofstream output;
	output.open(outName);

	for(int i = 0; i < ans.size(); i++){
		output << ans[i] << "\n";
	}
	output.close();
}