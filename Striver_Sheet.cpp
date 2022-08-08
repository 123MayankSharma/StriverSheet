 Day 1: Arrays
Q1)Set Matrix Zeros:
void setZeros(vector<vector<int>> &matrix){
	int n=matrix.size();
	int m=matrix[0].size();
	
	bool row=false,col=false;
	//check in first row and column for any original zeroes
	for(int i=0;i<n;i++){
		if(!matrix[i][0]){
			col=true;
		}
	}
	
	for(int j=0;j<m;j++){
		if(!matrix[0][j]){
			row=true;
		}
	}
	
	for(int i=1;i<n;i++){
		for(int j=1;j<m;j++){
			if(!matrix[i][j]){
				matrix[0][j]=0;
				matrix[i][0]=0;
			}
		}
	}
	
	for(int i=1;i<matrix.size();i++){
		if(matrix[i][0]==0){
			for(int j=0;j<m;j++){
				matrix[i][j]=0;
			}
		}
	}
	
	for(int j=1;j<m;j++){
		if(matrix[0][j]==0){
			for(int i=0;i<n;i++){
				matrix[i][j]=0;
			}
		}
	}
	
	if(row){
		for(int j=0;j<m;j++){
			matrix[0][j]=0;
		}
	}
	
	if(col){
		for(int i=0;i<n;i++){
			matrix[i][0]=0;
		}
	}
	
	
}
/************************************************************/

Q2)Pascals Triangle:
vector<vector<long long int>> generate(int n){
		vector<vector<long long int>> ans;
		ans.push_back({1});
		for(int i=1;i<n;i++){
			vector<long long int> tmp;
			long long int res=1;
			tmp.push_back(res);
			for(int j=0;j<i;j++){
				res=res*(i-j);
				res=res/(j+1);
				tmp.push_back(res);
			}
			ans.push_back(tmp);
		}
		
		return ans;
}
/************************************************************/

Q3)Kadanes Algorithm:
int maxSubArray(vector<int>& nums) {
		int ans=INT_MIN;
		int curr_sum=0;
    for(int i=0;i<nums.size();i++){
			curr_sum=curr_sum+nums[i];
			ans=max(ans,curr_sum);
			if(curr_sum<0){
				curr_sum=0;
			}
    }
    
    return ans;
}
/************************************************************/

Q4)Next Permutation:
void nextPermutation(vector<int>& a){
	
	int n=a.size();
	int pointAtWhichSortedSequenceEnds=-1;
	int indexOfNumberJustGreaterThanBreakingPointNumber=0;
	for(int i=n-2;i>=0;i--){
		if(a[i+1]>a[i]){
			pointAtWhichSortedSequenceEnds=i;
			break;
		}
	}
	
	//edge case in the case there is no break point
	if(pointAtWhichSortedSequenceEnds==-1){
		reverse(a.begin(),a.end());
	}else{
				for(int i=n-1;i>=0;i--){
		if(a[i]>a[pointAtWhichSortedSequenceEnds]){
			indexOfNumberJustGreaterThanBreakingPointNumber=i;
			break;
		}
	}
	
	swap(a[pointAtWhichSortedSequenceEnds],a[indexOfNumberJustGreaterThanBreakingPointNumber]);
	
	reverse(a.begin()+pointAtWhichSortedSequenceEnds+1,a.end());
	}

        
}
/************************************************************/

Q5) Best Time to Buy and Sell Stock:
int maxProfit(vector<int>& prices){
	int buyPrice=INT_MAX;
	int profit=0;
	int n=prices.size();
	for(int i=0;i<n;i++){
		buyPrice=min(buyPrice,prices[i]);
		profit=max(profit,prices[i]-buyPrice);
	}
	
	return profit;
}

/************************************************************/
Q6)Sort an array of 0 1 & 2:
void sort012(int *arr, int n)
{
   int lo=0,hi=n-1,mid=0;
   
   while(mid<=hi){
		                    
		if(a[mid]==1){
			mid++;
		}else if(a[mid]==0){
			swap(a[mid],a[lo]);
			mid++;
			lo++;
		}else if(a[mid]==2){
			//not incrementing mid here to check if after swapping
			//a 0 is obtained at mid so that it can be swapped with lo
			swap(a[mid],a[hi]);
			mid;
			hi--;
		}
   }
}

DAY1 DONE!
---------------------------------------------------------------------------------------------------------------
Day 2: Arrays Part-II

Q7)Rotate Matrix:
void rotate(vector<vector<int>>& matrix){
	//this is for n*n matrix
	int n=matrix.size();
	
	//transpose
	for(int i=0;i<n;i++){
		for(int j=0;j<i;j++){
			swap(matrix[i][j],matrix[j][i]);
		}
	}
	
	for(int i=0;i<n;i++){
		reverse(matrix[i].begin(),matrix[i].end());
	}
}

/************************************************************/

Q8)Merge Intervals:
vector<vector<int>> merge(vector<vector<int>>& intervals) {
vector<vector<int>> ans;

sort(intervals.begin(),intervals.end());

for(int i=0;i<intervals.size();i++){
		if(ans.size()==0){
				ans.push_back(intervals[i]);
		}else{
				//ans.back() represents last stored elemen 
				//i.e the previous interval
				//current ans vector
				
				
				//if the current interval has started
				//before the previous interval ended
				if(ans.back()[1]>=intervals[i][0]){
						
						//the end of the current interval needs to be maximum among the
						//2 overlapping intervals as we are merging them
						ans.back()[1]=max(ans.back()[1],intervals[i][1]);
				}else{
						ans.push_back(intervals[i]);
				}
				
		}
}
        
return ans;
}
/************************************************************/

Q9)Merge 2 Sorted Arrays:
 void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	
 //when nums1 size is n+m
	int i=m-1,j=n-1,k=m+n-1;
	
	while(i>=0 && j>=0)
	{
		if(nums1[i]>nums2[j]){
            nums1[k--]=nums1[i--];
		}else if(nums1[i]<nums2[j]){
				nums1[k--]=nums2[j--];
		}else{
				nums1[k--]=nums2[j--];
				nums1[k--]=nums1[i--];
		}
	}
			
	while(i>=0)
	nums1[k--]=nums1[i--];
			
	while(j>=0)
	nums1[k--]=nums2[j--];
	
	//when we have to put minimum numbers in nums1 and max numbers in nums 2
	void merge(int arr1[], int arr2[], int n, int m) {
	    // code here
	    int i=n-1,j=0;
	    
	    //send smaller elements from arr2 to arr1 and larger elements from arr1 to arr2
	    while(i>=0 && j<m){
	        if(arr1[i]>arr2[j]){
	            swap(arr1[i],arr2[j]);
	            i--;
	            j++;
	        }else{
	            i--;
	            j++;
	        }
	    }
	    
	    sort(arr1,arr1+n);
	    sort(arr2,arr2+m);
	}
    
}
/************************************************************/

Q10)Repeated and Missing Number:
int *findTwoElement(int *arr, int n) {
        // code here
      int *a=new int[2];
        
        for(int i=0;i<n;i++){
            if(arr[abs(arr[i])-1]>0){
                arr[abs(arr[i])-1]*=-1;
            }else{
                a[0]=abs(arr[i]); 
            }
        }
        
        for(int i=0;i<n;i++){
            if(arr[i]>0){
                a[1]=i+1;
            }
        }
        
        return a;
    }
/************************************************************/

Q11)Find repeating number
int findDuplicate(vector<int>& nums) {
	int n=nums.size();
	int slow=nums[0];
	int fast=nums[0];
	//using floyd cycle detection Algorithm
	do{
		slow=nums[slow];
		fast=nums[nums[fast]];
	}while(slow!=fast);
	
	slow=nums[0];
	while(slow!=fast){
		slow=nums[slow];
		fast=nums[fast];
	}
	
	return slow;
}

/********************************************************************/

Q12)Count Inversion
 #define ll long long
ll int merge(ll arr[], ll l, ll mid, ll r){
		ll n1 = mid - l +1;
		ll int inv = 0;
		ll n2 = r - mid;
		vector<ll> a(n1), b(n2);
		for(ll i=0;i<n1;i++){
				a[i] = arr[l+i];
		}
		for(ll i=0;i<n2;i++){
				b[i] = arr[mid+ 1+ i];
		}
		ll i = 0, j =0, k = l;
		while(i<n1 && j < n2){
				if(a[i] <=  b[j]){
						arr[k++] = a[i++];
				}
				else{
						arr[k] = b[j];
						inv += n1 - i;
						k++;
						j++;
						
				}
		}
		while(i < n1){
				arr[k++] = a[i++];
		}
		while(j < n2){
				arr[k++] = b[j++];
		}
		return inv;
}
    
ll int mergeSort(ll arr[], ll l, ll r){
		ll int inv = 0;
		if(l < r){
				int mid = l + (r- l)/2;
				inv += mergeSort(arr,l,mid);
				inv += mergeSort(arr,mid+1,r);
				
				inv += merge(arr,l,mid,r);
		}
		return inv;
}
    
long long int inversionCount(long long arr[], long long N)
{
		// Your Code Here
		ll int inv = mergeSort(arr,0,N-1);
		return inv;
}




DAY 2 DONE!!
------------------------------------------------------------------------------------------------
Day 3: Arrays Part-III

Q13)Search in a 2D matrix:
bool searchMatrix(vector<vector<int>>& matrix, int target){
	int i=0,j=matrix[0].size()-1;
	int n=matrix.size();
	//stand on the last element of first row
	//if target is smaller than curr_element move leftwards in the row
	//as row is sorted as last element is greatest
	//else move downwards as column is also sorted and the last element of
	//current row is smallest in the column we are standing in
	while(i<n && j>=0){
		if(matrix[i][j]==target){
			return 1;
		}else if(matrix[i][j]>target){
			j--;
		}else{
			i++;
		}
	}
	return 0;
}

//leetcode version
bool searchMatrix(vector<vector<int>>& matrix, int target){
  int n=matrix.size();
  int m=matrix[0].size();
  
  if(n==0){
		return 0;
  }
  
  int lo=0,hi=(n*m)-1;
  
  while(lo<=hi){
		int mid=(lo+(hi-lo)/2);
		//way to convert a 1d index to 2d index if row and column
		//size i.e n and m are given
		int row=mid/m;
		int col=mid%m;
		
		if(matrix[row][col]==target){
			return 1;
		}
		
		if(matrix[row][col]<target){
			lo=mid+1;
		}else{
			hi=mid-1;
		}
		
  }
  
  return false;
}

Q14)POW(X,n):
double myPow(double x, int n){
	double res=1.0; 
	long long nn=n;
	if(nn<0)nn*=-1;
	
	
	//when nn is even
	//x=x*x and n gets halved
	//else we multiply x by res and reduce power by 1;
	//(iterative)
	while(nn){
		if(nn%2){
			res=res*x;
			nn=nn-1;
		}else{
			x=x*x;
			nn=nn/2;
		}
	}
	
	if(n<0) res=(double)(1.0)/(double)(ans);
	return ans;
	
	
}

//Q14 recursive

double power(int x,long long nn){
	if(nn==0){
		return 1.0;
	}
	
	double tmp=power(x,nn/2);
	double res=tmp*tmp;
	
	if(nn%2) res*=x;
	
	return res;
}
double myPow(double x, int n){
	long long nn=n;
	if(nn<0) nn*=-1;
	double ans=power(x,nn);
	
	if(n<0) ans=(double)(1.0)/(double)(ans);
	return ans;
		
	
}

//Q14 with modulo
int power(long long x, unsigned int y,int p)
{
    int res = 1;     // Initialize result
 
     
  
    if (x == 0) return 0; // In case x is divisible by p;
 
    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res*x) % p;
 
        // y must be even now
        y = y>>1; // y = y/2
        x = (x*x) % p;
    }
    return res;
}
int modularExponentiation(int x, int n, int m) {
	// Write your code here.
    int ans=power(x,n,m);
    return ans;
}


Q15)Majority Element (>N/2 times)
int majorityElement(vector<int>& nums){
	int cnt=0,ele=0;
	int n=nums.size();
	for(int i=0;i<n;i++){
		if(cnt==0){
				ele=nums[i];
		}
		
		if(nums[i]==ele){
				cnt++;
		}else{
				cnt--;
		}
	}
        
  return ele;
}

Q16) MajorityElement (>N/3 times)
vector<int> majorityElement(vector<int>& nums){

	//one observation is that there can be at max 2 majority elements
	int num1=-1,num2=-1,count1=0,count2=0;
	int n=nums.size();
	for(int i=0;i<n;i++){
		if(nums[i]==num1){
			count1++;
		}else if(nums[i]==num2){
			count2++;
		}else if(count1==0){
			num1=nums[i];
			count1=1;
		}else if(count2==0){
			num2=nums[i];
			count2=1;
		}else{
			count1--;
			count2--;
		}
	}
	
	vector<int> ans;
	count1=0,count2=0;
	for(auto it:nums){
		if(it==num1)count1++;
		else if(it==num2)count2++;
	}
	
	if(count1>n/3){
		ans.push_back(num1);
	}
	
	if(count2>n/3){
		ans.push_back(num2);
	}
	
	return ans;
	
}

Q17)Grid Unique Paths:
int uniquePaths(int m, int n) {
	//one obvious solution is we can use dp but we will use combinatorics
	int a=m+n-2;
	int b=m-1;
	
	//to find = m+n-2Cm-1
	
	int ans=1;
	for(int i=1;i<=b;i++){
		res=res*(a-b+i);
		res=res/i;
	}
	
	return (int)res;
	
}

Q18)Reverse Pairs(leetcode)




DAY 3 not done as of now

------------------------------------------------------------------------------------------------
Day 4: Arrays Part-IV
Q19)2 sum:
vector<vector<int>> pairSum(vector<int> &nums, int s){
   // Write your code here.
   //using hashing
unordered_map<int,int> mp;
    vector<vector<int>> ans;
    for(auto it:nums){
        if(mp.count(s-it)){
            int count=mp[s-it];
            for(int i=0;i<count;i++){
                if(s-it>it){
                    ans.push_back({it,s-it});
                }else{
                    ans.push_back({s-it,it});
                }
            }
        }
        
        mp[it]++;
    }
    sort(ans.begin(),ans.end());
    return ans;
}



Q20)4sum
vector<vector<int>> fourSum(vector<int>& nums, int target) {
	vector<vector<int>> ans;
	
	if(nums.size()==0){
			return ans;
	}
	// we use 4 pointers here basically i,j,left and right
	//we use left and right to form 2 sum property and find target-(nums[i]+nums[j])
	int n=nums.size();
	sort(nums.begin(),nums.end());
	
	for(int i=0;i<n;i++){
		for(int j=i+1;j<n;j++){
				int tmp=target-nums[i]-nums[j];
				int left=j+1;
				int right=n-1;
				//applying 2 sum property
				while(left<right){
						if(nums[left]+nums[right]>tmp){
								right--;
						}else if(nums[left]+nums[right]<tmp){
								left++;
						}else{
								vector<int> t;
								t.push_back(nums[i]);
								t.push_back(nums[j]);
								t.push_back(nums[left]);
								t.push_back(nums[right]);
								ans.push_back(t);
								while(left<right && nums[left]==t[2]) left++;
								while(left<right && nums[right]==t[3]) right--;
								
						}
				}
				//avoiding duplicates
				while(j+1<n && nums[j+1]==nums[j]) ++j;
			}
					//avoiding duplicates
					while(i+1<n && nums[i+1]==nums[i]) ++i;
			
	}
	
	return ans;
}

Q21)Longest Consecutive sequence
 int longestConsecutive(vector<int>& nums) {
		if(nums.size()==0) return 0;
		unordered_set<int> st;
		for(int i=0;i<nums.size();i++){
				st.insert(nums[i]);
		}
		int ans=0;
		
		for(int i=0;i<nums.size();i++){
			//check if for a number x,x-1 exists in set or not
			//if it does then we can assume x to be the start of
			//a sequence and starting from x,we keep incrementing x and
			//keep checking if x+1 exists or not
			//and after counting length of sequence we update ans variable
				if(st.find(nums[i]-1)==st.end()){
						int tmp=1;
						int val=nums[i];
						while(st.find(val+1)!=st.end()){
								val+=1;
								tmp++;
						}
						ans=max(ans,tmp);
				}
		}
		
		return ans;
}

Q22)Count Number of Subarrays with given xor k
int subarraysXor(vector<int> &arr, int x)
{
    //    Write your code here.
	vector<int> xorArr(arr.size(),0);
	int ans=0;
	unordered_map<int,int> mp;
	xorArr[0]=arr[0];
	for(int i=1;i<arr.size();i++){
			xorArr[i]=xorArr[i-1]^arr[i];
	}
	
	for(int i=0;i<arr.size();i++){
			int tmp=x^xorArr[i];
			
			ans=ans+mp[tmp];
			if(xorArr[i]==x){
					ans++;
			}
			mp[xorArr[i]]++;
	}
    
  return ans;
}

Q23)Largest Subarray with 0 sum:
int maxLen(vector<int>&A, int n)
{   
		vector<int> prefix(n,0);
		prefix[0]=A[0];
		
		for(int i=1;i<n;i++){
				prefix[i]=prefix[i-1]+A[i];
		}
		int ans=0;
		
		unordered_map<int,int> mp;
		mp[0]=-1;
		for(int i=0;i<n;i++){
				if(mp.count(prefix[i])){
						ans=max(ans,i-mp[prefix[i]]);
				}else{
						mp[prefix[i]]=i;
				}
		}
		
		return ans;
}

Q24)Longest Substring without repeat:
int uniqueSubstrings(string S)
{
	//set based approach
	set<char> st;
	int n=S.size();
	int l=0,ans=0;
	
	for(int i=0;i<n;i++){
			//releasing characters until repeating element is
			//erased
			while(l<=i && st.find(S[i])!=st.end()){
				st.erase(S[l]);
				l++;
			}
			//updating answer continuously
			ans=max(ans,i-l+1);
			//acquiring elements
			st.insert(S[i]);
	}
	
	return ans;
}

int longestUniqueSubsttr(string S){
	//vector based fastest approach
	vector<int> v(26,0);
	int ans=0,l=0;
	
	for(int i=0;i<S.size();i++){
			int c=S[i]-'a';
			//update frequency
			v[c]++;
			//while frequency of current character is greater than 1
			//keep releasing the elements from left
			while(v[c]>1 && l<i){
					v[S[l++]-'a']--;
			}
			
			ans=max(ans,i-l+1);
	}
	
	return  ans;
        
    }
    
Day-4 DONE!!

------------------------------------------------------------------------------------------------
Day 5: Linked List
Q25 reverse a linked list
ListNode* reverseList(ListNode* head) {
			 ListNode* prev=NULL,*next=NULL,*curr=head;
        
        while(curr!=NULL){
            //first we backup the curr->next into next
            //before breaking link between curr and curr->next
            next=curr->next;
            //after breaking link with curr->next ,curr->next now points
            //to prev pointer
            curr->next=prev;
            //move the prev pointer to the current position
            //of curr
            prev=curr;
            //move curr forward
            curr=next;
            
            //repeat above steps until curr reaches end of list i.e 
            //all links have been reversed
        }
        
        //prev is the head of the reversed linked list
        return prev;
}

Q26) Find middle of linked list:
ListNode* middleNode(ListNode* head) {
	ListNode* slow=head,*fast=head;
	
		//since fast needs to jump like
		// fast=fast->next->next
		//we need to check if fast is not null and
		//fast->next is not null as fast->next->next being or not being null
		//doesnt pose an issue
		while(fast!=NULL && fast->next!=NULL){
			slow=slow->next;
			fast=fast->next->next;
		}
		
		return slow;
}

Q27)Merge 2 sorted Lists:
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if(!l1){
		return l2;
	}
	
	if(!l2){
		return l1;
	}
	
	//l1 will always be smaller list
	//l2 will always be larger list
	
	
	if(l1.val>l2.val){
		ListNode* tmp=l1;
		l1=l2;
		l2=tmp;
	}
	
	ListNode* ans=l1;
	
	while(l1!=NULL && l2!=NULL){
		ListNode* tmp=NULL;
		//while current node in l1 is smaller than l2
		//keep updating tmp as tmp is the last node of the current list l1 smaller than l2
		//that we have encountered till now
		//and then we move l1 forward
		while(l1!=NULL && l1.val<=l2.val){
			tmp=l1;
			l1=l1->next;
		}
		
		tmp->next=l2;
		
		//swap
		ListNode* temp=l1;
		l1=l2;
		l2=temp;
	}
	
	return ans;
}

//PepCoding code for merge 2 sorted linked list

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if(l1==NULL||l2==NULL) return l1!=NULL?l1:l2;
	
	ListNode* c1=l1;
	ListNode* c2=l2;
	
	ListNode* dummy=new ListNode(-1);
	ListNode* prev=dummy;
	
	while(c1!=NULL && c2!=NULL){
		if(c1->val<c2->val){
			prev->next=c1;
			c1=c1->next;
		}else{
			prev->next=c2;
			c2=c2->next;
		}
		
		prev=prev->next;
	}
	
	prev->next=c1!=NULL?c1:c2;
	
	return dummy->next;
	
}


Q28)Remove N-th node from back of LinkedList:
ListNode* removeNthFromEnd(ListNode* head, int n) {
	ListNode* dummy=new ListNode(-1);
	dummy->next=head;
	ListNode* slow=dummy;
	ListNode* fast=dummy;
	
	//creating a diffrence of n b/w slow and fast pointer
	while(n-- && fast!=null){
			fast=fast->next;
	}
	
	//moving fast to last element so that
	//slow ends up on nth element from the end
	while(fast->next!=NULL){
		slow=slow->next;
		fast=fast->next;
	}
	
	// ListNode* tmp=slow->next->next;
	ListNode* tmp1=slow->next;
	//deleting old link and making new link
	slow->next=slow->next->next;
	//deleting node from memory is optional
	delete(tmp1);
	
	return dummy->next;
}

Q29)Add two numbers represented as linked List in reverse order
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode* dummy=new ListNode();

	ListNode* tmp=dummy;
	int carry=0;
	while(l1!=NULL || l2!=NULL || carry){
			int sum=0;
			if(l1!=NULL){
					sum+=l1->val;
					l1=l1->next;
			}
			
			if(l2!=NULL){
					sum+=l2->val;
					l2=l2->next;
					
			}
			
			sum+=carry;
			carry=sum/10;
			ListNode* node=new ListNode(sum%10);
			tmp->next=node;
			tmp=tmp->next;
	}
	
	return dummy->next;
	
}

Q30) Delete Node in a Linked List
void deleteNode(ListNode* node) {
	node->val=node->next->val;
	node->next=node->next->next;
    
}

DAY 5 DONE!!!



------------------------------------------------------------------------------------------------
Day 6: Linked List Part-II

Q31) find intersection of linked list of Y shape
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
	
	//O(n) space complexity solution
	unordered_map<ListNode*,ListNode*> mp;
        
	ListNode* tmp=headA;
	
	while(tmp!=NULL){
		mp[tmp]=tmp;
		tmp=tmp->next;
	}
	
	
	tmp=headB;
	
	while(tmp!=NULL){
		if(mp.find(tmp)!=mp.end()){
			return mp[tmp];
		}
		tmp=tmp->next;
	}
	
	return NULL;   
}

int intersectPoint(Node* headA, Node* headB)
{
    // Your Code Here
    //O(2m) solution
        
        //find length of both nodes
        Node* tmp1=headA,*tmp2=headB;
        int c1=0,c2=0;
        while(tmp1!=NULL){
                c1++;
                tmp1=tmp1->next;
        }
        
        while(tmp2!=NULL){
                c2++;
                tmp2=tmp2->next;
            
        }
        
        //bring a pointer pointing to head of longer list
        //to same position as head of shorter list
        int ld=abs(c1-c2);
        Node* ans=c1>c2?headA:headB;
        
            for(int i=0;i<ld;i++){
                ans=ans->next;
            }
            
				//start iterating both pointers which point
				//to head of shorter list and the equivalent position
				//in longer list respectively
         tmp1=c1>c2?headB:headA;
        while(tmp1!=ans){
            tmp1=tmp1->next;
            // tmp2=tmp2->next;
            ans=ans->next;
        }
        
        
        return ans->data;
}

Q32)Detect cycle in Linked List
bool hasCycle(ListNode *head){
	
	ListNode *slow=head,*fast=head;
	
	while(fast!=NULL && fast->next!=NULL){
			
		slow=slow->next;
		fast=fast->next->next;
		if(slow==fast){
			return 1;
		}
	}
	
	return 0;
}
/************************************************************************************************/

Q33)Reverse a LinkedList in groups of size k.
ListNode* reverseKGroup(ListNode* head, int k) {
		ListNode* prev=NULL,*curr=head,*next=NULL;
		//reverse batch of k
		int c=0;
		//checking if batch of k exists or not
		int count=k;
		ListNode* tmp=head;
		int flag=0;
		for(int i=0;i<k;i++){
				if(tmp){
						tmp=tmp->next;
				}else{
						flag=1;
						break;
				}
		}
		
		if(flag){
				return head;
		}
		
		//reversing batch of k if it exists
		while(curr!=NULL && c<k){
				next=curr->next;
				curr->next=prev;
				prev=curr;
				curr=next;
				c++;
		}
		
		//the head pointer still points to the original head of our batch of
		//k elements
		//to connect 2 batches we connect head of 1 batch to the last element of         
		//another batch
		
		if(next!=NULL){
				head->next=reverseKGroup(next,k);
		}

		return prev;
}



/************************************************************************************************/
Q34)Check whether a linked list is palindrome?
ListNode* reverse(ListNode* head){
        ListNode* prev=NULL,*curr=head,*next=NULL;
        
        
        while(curr){
            next=curr->next;
            curr->next=prev;
            prev=curr;
            curr=next;
        }
        
        return prev;
}
 bool isPalindrome(ListNode* head) {
        ListNode* slow=head,*fast=head;
        
        while(fast->next && fast->next->next){
            slow=slow->next;
            fast=fast->next->next;
        }
        //reverse second half of linked list
        //so that last ith element of linked list can be compared 
        //to n-i th element of linked list without using extra space
        slow->next=reverse(slow->next);
        //make slow as head of the newly reversed seecond half of linked
        //list
        slow=slow->next;
        
        ListNode* tmp=head;
        
        while(slow!=NULL && tmp!=NULL){
            if(slow->val!=tmp->val){
                return false;
            }
            
            slow=slow->next;
            tmp=tmp->next;
        }
        return true;
    }

/**********************************************************************/
Q35)Find the starting Point of the loop of linked list
ListNode *detectCycle(ListNode *head) {
	ListNode* slow=head,*fast=head;
	if(!head || !head->next) return NULL;
	int flag=0;
	while(fast && fast->next){
			
			slow=slow->next;
			fast=fast->next->next;
			if(slow==fast){
					flag=1;
					break;
			}
	}
	
	if(!flag) return NULL;
	
	slow=head;
	while(slow!=fast && slow && fast){
			slow=slow->next;
			fast=fast->next;
	}
	
	return slow;
}

/*****************************************************************/
Q36 Flattening a linked List:
Node* merge(Node* a,Node* b){
	if(!a){
		return b;
	}
	
	if(!b){
		return a;
	}
	
	if(a->data>b->data){
		b->bottom=merge(b->bottom,a);
		return b;
	}else{
		a->bottom=merge(a->bottom,b);
		return a;
	}
	
}

Node *flatten(Node *root)
{
	Node* ans=NULL;
	Node* tmp=root;
	
	while(tmp!=NULL){
		ans=merge(ans,tmp);
		tmp=tmp->next;
	}
	
	return ans;
}


//using priority queue
struct mycomp {
    bool operator()(Node* a, Node* b)
    {
        return a->data > b->data;
    }
};
Node* flattenLinkedList(Node* head) 
{
	// Write your code here
     priority_queue<Node*, vector<Node*>, mycomp> p;
    while(head!=NULL){
        p.push(head);
        head=head->next;
    }
    Node* d=new Node(-1);
    Node* ans=d;
    while (!p.empty()) {
     
        Node* k = p.top();
        p.pop();
        ans->child=k;
        ans=ans->child;
        if (k->child)
            p.push(k->child);
    }
    
    return d->child;
}


//optimised recursive
Node *merge(Node *first, Node *second)
{
    // If the first is NULL return second
    if (first == NULL)
    {
        return second;
    }

    // If the second is NULL return first
    if (second == NULL)
    {
        return first;
    }

    Node *merged = NULL;

    if (first->data < second->data)
    {
        merged = first;
        merged->child = merge(first->child, second);
    }
    else
    {
        merged = second;
        merged->child = merge(first, second->child);
    }
    merged->next = nullptr;
    return merged;
}

Node *flattenLinkedList(Node *head)
{
    if (head == NULL || head->next == NULL)
    {
        return head;
    }

    // Recur on next node
    head->next = flattenLinkedList(head->next);

    // Merge with the current
    head = merge(head, head->next);
		
    return head;
}

------------------------------------------------------------------------------------------------
Day 7: Linked List and Arrays

Q37) Rotate LinkedList:
ListNode* rotateRight(ListNode* head, int k) {
		//for right rotation
		if(!head) return NULL;
		
		//count number of nodes
		int count=0;
		ListNode* tmp=head;
		
		while(tmp->next!=NULL){
				count++;
				tmp=tmp->next;
		}
		// if(!count)return NULL;
		count+=1;
		//make linked list a circulrly linked list
		tmp->next=head;
		
		//if a linked list of size x is rotated n*x times where n=integer 
		//then the linked list remains the same
		//so for any k actual number of rotations we need to
		// do is k%length
		k=k%count;
		
		//deattach length-k-1th node so that it's next node becomes head
		//i.e length-kth node
		ListNode* t=head;
		for(int i=0;i<count-k-1;i++){
				t=t->next;
		}
		
		ListNode* ans=t->next;
		t->next=NULL;
		return ans;
        
}

//for left rotation of linked list
Node* reverse(Node* head){
        Node* prev=NULL,*curr=head,*next=NULL;
        
        while(curr!=NULL){
            next=curr->next;
            curr->next=prev;
            prev=curr;
            curr=next;
        }
        
        return prev;
    }
    Node* rotate(Node* head, int k)
    {
        // Your code here
        if(!head) return NULL;
        
        //count number of nodes
        int count=0;
        head=reverse(head);
        Node* tmp=head; 
                          
        
        while(tmp->next!=NULL){
            count++;
            tmp=tmp->next;
        }
        // if(!count)return NULL;
        count+=1;
        //make linked list a circulrly linked list
        tmp->next=head;
        
        //if a linked list of size x is rotated n*x times where n=integer 
        //then the linked list remains the same
        //so for any k actual number of rotations we need to
        // do is k%length
        k=k%count;
        
        //deattach length-k-1th node so that it's next node becomes head
        //i.e length-kth node
        Node* t=head;
        for(int i=0;i<count-k-1;i++){
            t=t->next;
        }
        
       Node* ans=t->next;
        t->next=NULL;
        ans=reverse(ans);
        return ans;
}
/************************************************************************/
Q38)Clone a Linked List with next and random pointer:
	//o(n) space solution
	Node* copyRandomList(Node* head) {
		unordered_map<Node*,Node*> mp;
		
		Node* tmp=head;
		//make a copy of every node in original linked list
		//and map copy of every node with itself
		while(tmp!=NULL){
				Node* t=new Node(tmp->val);
				mp[tmp]=t;
				tmp=tmp->next;
		}
		
		tmp=head;
		while(tmp!=NULL){
			//next pointer in deep copy's any given node
			//is the copy of next pointer of that original node
			//and random pointer of any node in deep copy
			//is the copy of random pointer in the original node
			//which can be obtained from the map
				mp[tmp]->next=mp[tmp->next];
				mp[tmp]->random=mp[tmp->random];
				tmp=tmp->next;
		}
		
		return mp[head];
	}


//O(1) solution
Node *copyList(Node *head)
{
		 if (head == NULL) return NULL;
        
        //step1: Add the copy nodes in between original list
        Node *curr = head;
        while (curr != NULL) {
            Node *next = curr->next;
            Node *temp = new Node(curr->data);
            temp->next = next;
            curr->next = temp;
            curr = next;
        }
        
        //step2: Assign arbitrary pointers for the copy nodes using original list
        curr=head;
        while (curr != NULL) {
            if (curr->arb == NULL) {
                curr->next->arb = NULL;
            }
            else {
                curr->next->arb = curr->arb->next;
            }
            
            curr=curr->next->next;
        }
        
        //step3: connect the copy nodes together and remove them from original list
        curr=head;
        Node *head2 = curr->next;
        while (curr != NULL) {
            Node *next = curr->next->next;
            if (next != NULL) {
                curr->next->next = next->next;    
            }
            else {
                curr->next->next = next;
            }
            
            curr->next=next;
            curr=next;
        }
        
        return head2;
}

Q39)3 sum:
vector<vector<int>> threeSum(vector<int>& nums) {
	//2 pointer approach has been used here
    vector<vector<int>> ans;
    if(nums.size()<3)return ans;
    sort(nums.begin(),nums.end());
		for(int i=0;i<nums.size()-2;i++){
			if(i==0||i>0 && nums[i]!=nums[i-1]){
				int lo=i+1,hi=nums.size()-1,sum=0-nums[i];
				while(lo<hi){
					if(nums[lo]+nums[hi]==sum){
						ans.push_back({nums[i],nums[lo],nums[hi]});
						//while loops to avoid duplicates
						while (lo < hi && nums[lo] == nums[lo+1]) lo++;
            while (lo < hi && nums[hi] == nums[hi-1]) hi--;
						lo++;
						hi--;
					}else if(nums[lo]+nums[hi]<sum){
						lo++;
					}else{
						hi--;
					}
				}
			}
		}
		return ans;
}

Q40)Trapping Rainwater:
int trap(vector<int>& height){
	int ans=0;
	int l=0,r=height.size()-1;
	int lmax=0,rmax=0;
	while(l<r){
		lmax=max(lmax,height[l]);
		rmax=max(rmax,height[r]);
		
		if(lmax<rmax){
			//if at current point highest tower on left 
			//is smaller than highest tower on right
			//it simply means we can fill water = lmax-current_tower's height
			ans=ans+lmax-height[l];
			l++;
		}else{
			//the above logic also applies if highest tower on right
			//is smaller than highest tower on left then at current point on r
			//we can fill water = rmax-current_tower's height
			ans=ans+rmax-height[r];
			r--;
		}
	}
	
	return ans;
        
}

Q41)Remove Duplicate from Sorted array:
int removeDuplicates(vector<int>& nums) {
	int ans=0;
	int i=0;
	while(i<nums.size()){
			while(i<nums.size()-1 && nums[i]==nums[i+1]){
					i++;
			}
			swap(nums[ans],nums[i]);
			i++;
			ans++;
	}
	return ans;
}

Q42)Max Consecutive Ones:
int findMaxConsecutiveOnes(vector<int>& nums) {
	 int ans=0;
	int i=0;
	while(i<nums.size()){
		if(nums[i]==1){
					int tmp=1;
					while(i<nums.size()-1 && nums[i]==nums[i+1]){
							tmp++;
							i++;
					}
					ans=max(ans,tmp);
					i++;
			}else{
					i++;
			}
	}
	
	return ans;
        
}


------------------------------------------------------------------------------------------------
Day 8: Greedy Algorithm

Q43)N meetings in one room:
int maxMeetings(int start[], int end[], int n)
{
	// Your code here
	vector<pair<int,int>> vp;
	for(int i=0;i<n;i++){
			vp.push_back({start[i],end[i]});
	}
	
	//sorting meetings based n their ending time
	//as if we do all the meetings which ended earlier
	//we will be able to do maximum number of meetings
	sort(vp.begin(),vp.end(),[](pair<int,int> a,pair<int,int> b){
			return a.second<b.second;
	});
	
	int prevEnd=vp[0].second;
	int count=1;
	for(int i=1;i<n;i++){
			//if the current meeting started after 
			//previous meeting ended
			//we can perform that meeting
			//and after doing the meeting we update the
			//ending time of meeting in prevEnd to compare it
			//to the start of the next meeting
			if(vp[i].first>prevEnd){
					prevEnd=vp[i].second;
					count++;
			}
	}
		
	return count;
}

Q44)Minimum number of platforms required for a railway:
int findPlatform(int arr[], int dep[], int n){
	//sort arrival and departure
	sort(arr,arr+n);
	sort(dep,dep+n);
	
	//what we are maintainig here is the max number of trains
	//at a particular time
	
	int i=0;//pointing at arrival time
	int j=0;//ponting at departure time
	
	int maxTrain=0;
	int platform=0;
	
	while(i<n && j<n){
		if(arr[i]<=dep[j]){
			//when one train is entering before another train leaves
			maxTrain++;
			i++;
		}else{
			//if a train is arriving after one train has departed 
			//i.e one platform has become empty
			//and the number of trains present at the same time has reduced by 1
			maxTrain--;
			j++;
		}
		
		platform=max(platform,maxTrain);
	}
	
	return platform;
	
}

Q45)Job scheduling:
 vector<int> JobScheduling(Job arr[], int n) {
  
        // your code here
	vector<int> ans(2,0);
	sort(arr,arr+n,[](Job a,Job b){
		return	a.profit>b.profit;
	});
	
	//we greedily do the job with max profit first
	//and do every job that we have on the last day possible
	//so that on other days we can do other jobs
	int maxDeadline=-1;
	
	for(int i=0;i<n;i++){
			maxDeadline=max(maxDeadline,arr[i].dead);
	}
	
	vector<int> daysWhenTaskPerformed(maxDeadline+1,-1);
	int count=0,profit=0;
	
	for(int i=0;i<n;i++){
			
			for(int j=arr[i].dead;j>=1;j--){
				//a free day is found
					if(daysWhenTaskPerformed[j]==-1){
						daysWhenTaskPerformed[j]=i;
						count++;
						profit+=arr[i].profit;
						break;
					}
			}
	}
	ans[0]=count;
	ans[1]=profit;
	return ans;
        
} 

Q46)fractional Knapsack:
double fractionalKnapsack(int W, Item arr[], int n)
{
		// Your code here
		double ans=0;
		sort(arr,arr+n,[](Item a,Item b){
				double valuePerUnitweighta=double(a.value)/(double)(a.weight);
				double valuePerUnitweightb=double(b.value)/(double)(b.weight);
				
				return valuePerUnitweighta>valuePerUnitweightb;
				
		});
		
		int curWeight=0;
		
		for(int i=0;i<n;i++){
			if(curWeight+arr[i].weight<=W){
				curWeight+=arr[i].weight;
				ans+=arr[i].value;
			}else{
				int remainingWeight=W-curWeight;
				ans+=(double)(arr[i].value)/(double)(arr[i].weight)*remainingWeight;
				break;
			}
		}
		return ans;
		
}


Q47)Greedy algorithm to find minimum number of coins:
int findMinimumCoins(int V) 
{
    // Write your code here
    vector<int> coins={1, 2, 5, 10, 20, 50, 100, 500, 1000};
        int ans=0;
         for (int i = coins.size() - 1; i >= 0; i--) {
            while (V >= coins[i]) {
              V -= coins[i];
              ans++;
            }
      }
        
        return ans;
}
Q48)Activity Selection(same as N meetiings in one room)
------------------------------------------------------------------------------------------------
Day 9: Recursion
Q49)Subset Sums:
void util(vector<int> &arr,int sum,int N,int i,vector<int> &ans){
	if(i==N){
			ans.push_back(sum);
			return;
	}
	
	util(arr,sum,N,i+1,ans);
	util(arr,sum+arr[i],N,i+1,ans);
	
}

vector<int> subsetSums(vector<int> arr, int N)
{
		// Write Your Code here
		vector<int> ans;
		util(arr,0,N,0,ans);
		
		return ans;
}

Q50)return all possible subset of array:
void util(set<vector<int>> &ans,int i,vector<int> tmp,vector<int> &nums){
	if(i==nums.size()){
		sort(tmp.begin(),tmp.end());
		ans.insert(tmp);
		return;
	}
	
	tmp.push_back(nums[i]);
	util(ans,i+1,tmp,nums);
	tmp.pop_back();
	util(ans,i+1,tmp,nums);
}	

vector<vector<int>> subsetsWithDup(vector<int>& nums){
	set<vector<int>> ans;
	vector<int> tmp;
	util(ans,0,tmp,nums);
	vector<vector<int>> v;
	for(auto it:ans){
		v.push_back(it);
	}
	return v;
}


Q51)Combination sum 1:
void func(int ind, vector<vector<int>> &ans, vector<int> &temp, int B, vector<int> &A){
	if(ind==(int)A.size()){
			if(B==0){
			ans.push_back(temp);
			}
			return;
	}
	if(A[ind]<=B){
			temp.push_back (A[ind]);
			func(ind,ans,temp,B-A[ind],A);
			temp.pop_back();
	}
	func(ind+1,ans,temp,B,A);
}
vector<vector<int>> combinationSum(vector<int>& A, int B) {
  vector<vector<int>> ans;
	set<int> st;
	vector<int> temp;
	for(int i=0; i<A.size(); i++)
	st.insert(A[i]);
	while(A.size()>0)A.pop_back();
	for(auto it:st)
	A.push_back(it);
	sort(A.begin(),A.end());
	func(0,ans,temp,B,A);
	return ans;
}

Q52)Combination Sum 2:
void solve(int idx, vector <int> &a, int b, vector <int> temp,vector<vector<int>> &res){
	if(b == 0){
			res.push_back(temp);
			return;
	}
	if(idx == a.size())return;
	if(b < 0)return;
	sort(a.begin(), a.end());
	for(int i = idx; i < a.size(); i++){
		if(i > idx && a[i] == a[i-1])continue;
		temp.push_back(a[i]);
		solve(i + 1, a, b - a[i], temp,res);
		temp.pop_back();
	}
}
vector<vector<int>> combinationSum2(vector<int> &a, int b) {
	vector<vector<int>> res;
	vector <int> temp;
	solve(0, a, b, temp,res);
	return res;
}

Q53)Palindrome Partitioning:
bool ispal(string t){
	int i=0,j=s.size()-1;
	
	while(i<j){
		if(t[i]==t[j]){
			i++;
			j--;
		}else{
			return false;
		}
	}
	return true;
}

void util(string s,int idx,vector<string> &tmp,vector<vector<string>> &ans){
	if(idx==s.size()){
		ans.push_back(tmp);
		return;
	}
	
	for(int i=idx;i<s.size();i++){
		string t=s.substr(idx,(i-idx)+1);
		if(ispal(t)){
			tmp.push_back(t);
			util(s,i+1,tmp,ans);
			tmp.pop_back();
		}
	}
}
vector<vector<string>> partition(string s) {
	//eg-abaaba
	vector<vector<string>> ans;
	vector<string> tmp;
	util(s,0,tmp,ans);
	return ans;
}

Q54)Kth permutation sequence:
 string getPermutation(int n, int k) {
	//brute force
		vector<int> v;
		for(int i=1;i<=n;i++){
				v.push_back(i);
		}
		sort(v.begin(),v.end());
		for(int i=0;i<k-1;i++){
				next_permutation(v.begin(),v.end());
		}
		
		string s="";
		for(auto it:v){
				s.push_back(it+'0');
		}
			
		return s;
}
	
//optimised
string getPermutation(int n, int k) {

	int fact=1;
	vector<int> numbers;
	for(int i=1;i<n;i++){
		//we compute (n-1)!
		fact=fact*i;
		//store all numbers in numbers vector
		numbers.push_back(i);
	}
	numbers.push_back(n);
	string ans="";
	//we follow zero based indexing so we reduce k by 1
	k=k-1;
	
	while(1){
		//the kth permutation starts with the given number
		ans.push_back(numbers[k/fact]+'0');
		numbers.erase(numbers.begin()+k/fact);
		if(numbers.size()==0) break;
		//number of numbers we need to arrange
		k=k%fact;
		fact=fact/numbers.size();
	}
	return ans;
}

------------------------------------------------------------------------------------------------

Day 10: Recursion and Backtracking
Q55)Permutation of string/Array:
void util(vector<int> &nums,int i,vector<vector<int>> &ans){
	if(i==nums.size()){
			ans.push_back(nums);
			return ;
	}
	
	for(int j=i;j<nums.size();j++){
			swap(nums[i],nums[j]);
			util(nums,i+1,ans);
			swap(nums[i],nums[j]);
	}
}
vector<vector<int>> permute(vector<int>& nums) {
		vector<vector<int>> ans;
		util(nums,0,ans);
		return ans;
		
}

Q56)N queens problem:
bool isSafe(vector<string> tmp,int x,int y){
	for(int i=0;i<x;i++){
		for(int j=0;j<tmp.size();i++){
			if(tmp[i][j]=='Q'){
				//checking that if a queen is not present 
				//in the same column as the current queen 
				//or if there is a queen in the diagonal in
				//which the current queen is present
				if(y==j || abs(x-i)==abs(y-j)){
					return false;
				}
			}
		}
	}
	return true;
}
void util(int x,vector<string> &tmp,int n,vector<vector<string>> &ans){
	if(x==n){
		ans.push_back(tmp);
		return;
	}
	
	for(int y=0;y<n;y++){
		if(isSafe(tmp,x,y)){
			tmp[x][y]='Q';
			util(x+1,tmp,n,ans);
			tmp[x][y]='.';
		}
	}
}
vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> ans;
	
	vector<string> tmp(n,string(n,'.'));
	util(0,tmp,n,ans);
	return ans;
}

Q57)Sudoku Solver

Q58)M coloring problePm:

bool isSafe(int node,int color[],bool graph[101][101],int n,int col){
	for(int k=0;k<n;k++){
		//if any of node from 0 to n-1
		//having edge with node have same color
		//return false;
		if(k!=node && graph[k][node]==1 && color[k]==col){
			return 0;
		}
	}
	
	return 1;
}
bool solve(int node,int color[],int m,int n,bool graph[101][101]){
	if(node==n){
		return true;
	}
	
	for(int i=1;i<=m;i++){
		if(isSafe(node,color,graph,n,i)){
			color[node]=i;
			if(solve(node+1,color,m,n,graph))return true;
			color[node]=0;
		}
	}
	
	return false;
}
bool graphColoring(bool graph[101][101], int m, int n) {
    // your code here
    int color[n]={0};
    if(solve(0,color,m,n,graph)) return true;
    return false;
}

Q59)Rat in a maze:
  
void util(vector<string> &ans,string tmp,vector<vector<int>> &m,int n,int i,int j,vector<vector<int>> &vis){
		if(i<0 || i>=m.size() || j<0 || j>=m[0].size()||vis[i][j]==1 || m[i][j]==0){
				return;
		}
		
		if(i==n-1 && j==n-1){
				ans.push_back(tmp);
				return;
		}
		
		vis[i][j]=1;
		
		vector<char> dir={'U','L','D','R'};
		
		vector<int> row={-1,0,1,0};
		vector<int> col={0,-1,0,1};
		
		for(int k=0;k<row.size();k++){
				
				util(ans,tmp+dir[k],m,n,i+row[k],j+col[k],vis);
				
		}

		vis[i][j]=0;
}
vector<string> findPath(vector<vector<int>> &m, int n) {
		// Your code goes here
		vector<string> ans;
		
		
		string tmp="";
		vector<vector<int>> vis(n,vector<int>(n,0));
		util(ans,tmp,m,n,0,0,vis);
		//sort(ans.begin(),ans.end());
		return ans;
}

Q60)Word Break:
void util(string s,vector<string> &dict,vector<string> &ans,string tmp){
    if(s.size()==0){
        ans.push_back(tmp);
        return;
    }
    for(int i=0;i<s.size();i++){
        string left=s.substr(0,i+1);
        if(find(dict.begin(),dict.end(),left)!=dict.end()){
            string right=s.substr(i+1);
            util(right,dict,ans,tmp+left+" ");
        }
    }
}
vector<string> wordBreak(string &s, vector<string> &dictionary)
{
    // Write your code here
            vector<string> ans;
          util(s,dictionary,ans,"");
          return ans;
}
------------------------------------------------------------------------------------------------
Day 11: Binary Search
Q61)Nth root of a number:
double findNthRootOfM(int m, long long n){
	
	// Write your code h
	double lo=1;
	double hi=n;
	
	while(hi-lo>1e-7){
		double mid=(lo+hi/2.0);
		long long ans=1;
		for(int i=0;i<m;i++){
			ans=ans*mid;
		}
		if(ans<n){
			lo=mid;
		}else{
			hi=mid;
		}
	}
	return lo;
}

Method 2:Newton raphson method
double findNthRootOfM(int n, long long m) {

    // Variable to store maximum possible error in order
    // to obtain the precision of 10^(-6) in the answer.
    double error = 1e-7;

    // Difference between the current answer, and the answer
    // in next iteration, which we take as big as possible initially.
    double diff = 1e18;

    // Guessed answer value
    double xk = 2;

    // We keep on finding the precise answer till the difference between
    // answer of two consecutive iteration become less than 10^(-7).
    while (diff > error) {

        // Answer value in the next iteration.
        double xk_1 = (pow(xk, n) * (n - 1) + m) / (n * pow(xk, n - 1));

        // Difference of answer in consecutive states updated.
        diff = abs(xk - xk_1);

        // Updating the current answer with the answer of next iteration.
        xk = xk_1;
    }

    // Returning the nthRootOfM with precision upto 6 decimal places
    // which is xk.
    return xk;
}

Q62)find median in row-wise sorted matrix(n*m%2!=0):
int median(vector<vector<int>> &matrix, int r, int c){
	// code here   
	int lo=0,hi=1e9;
	int n=r*c;
	while(lo<=hi){
			int mid=(lo+hi)/2;
			//counter for counting number
			//of values smalller than current mid as
			//median is a number which has n/2 numbers smaller
			//than itself and n/2 numbers greater than itself
			//in case of an array or matrix
			int lesserValues=0;
			for(int i=0;i<r;i++){
					int l=0,h=c-1;
					while(l<=h){
							int m=l+(h-l)/2;
							if(matrix[i][m]<=mid)l=m+1;
							else h=m-1;
					}
					lesserValues+=l;
			}
			if(lesserValues<=n/2){
					lo=mid+1;
			}else{
					hi=mid-1;
			}
	}
	
	return lo;
}

Q63)Find the element that appears once in a sorted array, and the rest element appears twice:
int singleNonDuplicate(vector<int>& nums){
	/*
		in this left array, the first instance of 
		every element is occurring on the even index 
		and the second instance on the odd index. 
		Similarly in the right array, the first 
		instance of every element is occurring on the
		odd index and the second index is occurring on 
		the even index.
		This is summarized below.
		
		We will check our mid element, if it is in the 
		left array, we will shrink our left array to 
		the right of this mid element, 
		if it is in the right array, we will shrink 
		the right array to the left of this mid element. 
		This binary search process will continue 
		till the right array surpasses our left one and 
		low is pointing towards the breakpoint.
		
		
	*/
   int n=nums.size(); 
   int low = 0;
	 int high = n - 2;

		while (low <= high) {
			int mid = (low + high) / 2;
			
			if (mid % 2 == 0) {
					if (nums[mid] != nums[mid + 1]) 
					//Checking whether we are in right half

							high = mid - 1; //Shrinking the right half
					else
							low = mid + 1; //Shrinking the left half
			} else {

					//Checking whether we are in right half
					if (nums[mid] == nums[mid + 1]) 
							high = mid - 1; //Shrinking the right half
					else
							low = mid + 1; //Shrinking the left half
			}
		}

		return nums[low];
}


Q64)Search element in rotated sorted array:
int search(vector<int>& a, int target){
	 int lo = 0, hi = a.size() - 1;
	while(lo<=hi){
		int mid=(lo+hi)/2;
		
		if(a[mid]==target) return mid;
		if(a[low]<=a[mid]){
			//if left part is sorted and target
			//lies in left part
			if(target>=a[lo] && target<=a[mid]){
				high=mid-1;
			}else{
				lo=mid+1;
			}
		}else{
			if(a[hi]>a[mid]){
				//if right part is sorted and target
			//lies in right part
				if(target>=a[mid] && target<=a[hi]){
					lo=mid+1;
				}else{
					hi=mid-1;
				}
			}
		}
	}
	
   return -1;     
}

Q65)Median of 2 sorted array of diffrent size:
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2){
	//will do but pending for now
        
}



Q66)kth element of 2 sorted arrays
int kthElement(int arr1[], int arr2[], int n, int m, int k)
{
	if(n>m){
		return kthElement(arr2,arr1,m,n,k);
	}
	
	int low=max(0,k-m),high=min(k,n);
	
	while(low<=high){
		int cut1=(low+high)>>1;
		int cut2=k-cut1;
		int l1=cut1==0?INT_MIN:arr1[cut1-1];
		int l2=cut2==0?INT_MIN:arr2[cut2-1];
		int r1=cut1==n?INT_MAX:arr1[cut1];
		int r2=cut2==m?INT_MIN:arr2[cut2];
		
		if(l1<=r2 && l2<=r1){
			return max(l1,l2); 
		}
		
		else if(l1>r2){
			high=cut1-1;
		}
		else{
			low=cut1+1;
		}
		
	}
		return 1;
}
------------------------------------------------------------------------------------------------
Day 12: Heaps
Q)Find Median from Data Stream
class MedianFinder {
//we are splitting stream into roughly 
//2 halves.
//we take one min and one max heap
//if both heaps are empty we store
//incoming element in max heap
//else we compare whether current element is
//smaller than top of max heap. if yes
//element goes to max heap else element goes to min heap.
//then we check if mnh.size()-mxh.size()<=1
//else we shift top element from min heap
//to max heap.
//if mxh.size()==mnh.size() then median is
//average of top elements of both heap
//else median is top element of the larger heap.
public:
	priority_queue<int> mxh;
	priority_queue<int,vector<int>,greater<int>> mnh;
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        if(mxh.empty()||mxh.top()>num){
					mxh.push(num);
        }else{
					mnh.push(num);
        }
        
        if(mxh.size()>mnh.size()+1){  
					mnh.push(mxh.top());
					mxh.pop();
        }else if(mnh.size()>mxh.size()+1){
					mxh.push(mnh.top());
					mnh.pop();
        }
    }
    
    double findMedian() {
        if(mnh.size()==mxh.size()){
					return (mxh.top()+mnh.top())/2.0;
        }else{
					if(mxh.size()>mnh.size()){
						return mxh.top();
					}else{
						return mnh.top();
					}
        }
    }
};

Q)Find K max pair sum Combination
vector<int> kMaxSumCombination(vector<int> &A, vector<int> &B, int n, int K){
	// Write your code here.
    
    //what we are roughly doing is
    //we want to take the elements from top of array i.e maximum
    //elements and pair them up in such a way that we get top k sum pair.
    
    //we can do this by comparing greatest element  we have in pq from a
    //with the next greatest element we have in pq from b and vice versa.
    //what this will do is make sure that we are always getting max sum pair on top
    //as we are considering all possibilities and using max heap so that 
    //max_sum_pair comes on top.
     sort(A.begin(), A.end()); 
    sort(B.begin(), B.end()); 
  
    int N = A.size(); 
  
    // Max heap which contains tuple of the format 
    // (sum, (i, j)) i and j are the indices  
    // of the elements from array A 
    // and array B which make up the sum. 
    priority_queue<pair<int, pair<int, int> > > pq; 
  
    // my_set is used to store the indices of  
    // the  pair(i, j) we use my_set to make sure 
    // the indices doe not repeat inside max heap. 
    set<pair<int, int> > my_set; 
  
    // initialize the heap with the maximum sum 
    // combination ie (A[N - 1] + B[N - 1]) 
    // and also push indices (N - 1, N - 1) along  
    // with sum. 
    pq.push(make_pair(A[N - 1] + B[N - 1], 
                      make_pair(N-1, N-1))); 
  
    my_set.insert(make_pair(N - 1, N - 1)); 
  
    // iterate upto K 
    vector<int> ans;
    for (int count=0; count<K; count++) { 
  
        // tuple format (sum, (i, j)). 
        pair<int, pair<int, int> > temp = pq.top(); 
        pq.pop(); 
  
        ans.push_back(temp.first);
  
        int i = temp.second.first; 
        int j = temp.second.second; 
  
        int sum = A[i - 1] + B[j]; 
  
        // insert (A[i - 1] + B[j], (i - 1, j))  
        // into max heap. 
        pair<int, int> temp1 = make_pair(i - 1, j); 
  
        // insert only if the pair (i - 1, j) is  
        // not already present inside the map i.e. 
        // no repeating pair should be present inside  
        // the heap. 
        if (my_set.find(temp1) == my_set.end()) { 
            pq.push(make_pair(sum, temp1)); 
            my_set.insert(temp1); 
        } 
  
        // insert (A[i] + B[j - 1], (i, j - 1))  
        // into max heap. 
        sum = A[i] + B[j - 1]; 
        temp1 = make_pair(i, j - 1); 
  
        // insert only if the pair (i, j - 1) 
        // is not present inside the heap. 
        if (my_set.find(temp1) == my_set.end()) { 
            pq.push(make_pair(sum, temp1)); 
            my_set.insert(temp1); 
        } 
    } 
    
    return ans;
}
------------------------------------------------------------------------------------------------
Day 13: Stack and Queue
------------------------------------------------------------------------------------------------
Day 14: Stack and Queue Part-II
------------------------------------------------------------------------------------------------
Day 15: String
------------------------------------------------------------------------------------------------
Day 16: String Part-II
------------------------------------------------------------------------------------------------
Day 17: Binary Tree
Q)Bottom View
vector<int> bottomView(BinaryTreeNode<int> * root){

    // Write your code here.
    map<int,int> mp;
    vector<int> ans;
    if(!root){
        return ans;
    }
    
    queue<pair<BinaryTreeNode<int>*,int>> q;
    q.push({root,0});
    while(!q.empty()){
        BinaryTreeNode<int>* node=q.front().first;
        int x=q.front().second;
        q.pop();
        
        mp[x]=node->data;
        if(node->left){
            q.push({node->left,x-1});
        }
        
        if(node->right){
            q.push({node->right,x+1});
        }
        
        
    }
    
    for(auto it:mp){
        ans.push_back(it.second);
    }
    return ans;
    
}

Q)top view
#include<bits/stdc++.h>
vector<int> getTopView(TreeNode<int> *root) {
    // Write your code here.

    // Write your code here.
    map<int,int> mp;
    vector<int> ans;
    if(!root){
        return ans;
    }
    
    queue<pair<TreeNode<int>*,int>> q;
    q.push({root,0});
    while(!q.empty()){
			
        TreeNode<int>* node=q.front().first;
        int x=q.front().second;
        q.pop();
        
        if(mp.find(x)==mp.end())
        mp[x]=node->val;
        
        if(node->left){
            q.push({node->left,x-1});
        }
        
        if(node->right){
            q.push({node->right,x+1});
        }
        
        
    }
    
    for(auto it:mp){
        ans.push_back(it.second);
    }
    return ans;
    


}

Q3) Verical view:
 vector<vector<int>> verticalTraversal(TreeNode* root) {
		map<int,map<int,multiset<int>>> mp;
		queue<pair<TreeNode*,pair<int,int>>> q;
		q.push({root,{0,0}});
		
		while(!q.empty()){
				auto p=q.front();
				q.pop();
				TreeNode* node=p.first;
				int x=p.second.first;
				int y=p.second.second;
				mp[x][y].insert(node->val);
				
				
				if(node->left){
						q.push({node->left,{x-1,y+1}});
				}
				
				if(node->right){
						q.push({node->right,{x+1,y+1}});
				}
		}
		
		vector<vector<int>> ans;
		
		for(auto p:mp){
				vector<int> col;
				for(auto q:p.second){
						col.insert(col.end(),q.second.begin(),q.second.end());
				}
				ans.push_back(col);
		}
		return ans;
}
 
Q)Max Width of Binary Tree   
int widthOfBinaryTree(TreeNode* root) {
 int ans = 0;
queue<pair<TreeNode* , int>> q;
q.push({root , 0});
//creating index for every node
	//so Width is index of last element of a given level
	//minus index of first element of a given level.
	
 //index of right node=2*(idx of parent)+2
 //index of left node=2*(idx of parent)+1
  
 while(!q.empty()) {
		vector<pair<TreeNode* , int>> v;
		int sz = q.size();
		while(sz--) {
			pair<TreeNode* , int> p = q.front();
			q.pop();
			TreeNode* node = p.first;
			long long num = p.second;
			v.push_back(p);
			if(node -> left != NULL) q.push({node -> left , 2*num + 1});
			if(node -> right != NULL) q.push({node -> right , 2*num + 2});
		}
	int minm = v[0].second , maxm = v[(int)v.size()-1].second;
	ans = max(maxm - minm + 1 , ans);
}
return ans;
}

------------------------------------------------------------------------------------------------
Day 18: Binary Tree part-II
Q)height of tree
int height(Node* root){
	if(!root) return 0;
	int lh=height(root->left);
	int rh=height(root->right);
	return 1 + max(lh,rh);
}

Q)Diameter of binary tree
int util(TreeNode* root){
		if(!root) return 0;
		
		int lh=util(root->left);
		int rh=util(root->right);
		
		return 1+max(lh,rh);
}
void preorder(TreeNode* root,int &ans){
		if(!root){
				return;
		}
		
		int lh=util(root->left);
		int rh=util(root->right);
		ans=max(ans,lh+rh);
		
		preorder(root->left,ans);
		preorder(root->right,ans);
}
int diameterOfBinaryTree(TreeNode* root) {
		int ans=0;
		preorder(root,ans);
		
		return ans;
}

Q)check for height balanced binary tree
int dfs(BinaryTreeNode<int> *root){
    if(!root){
        return 0;
    }
    
    int lh=dfs(root->left);
    if(lh==-1)return -1;
    int rh=dfs(root->right);
    if(rh==-1)return -1;
    
    if(abs(lh-rh)>1)return -1;
    
    return 1+max(lh,rh);
}
bool isBalancedBT(BinaryTreeNode<int>* root) {
    // Write your code here.
    return dfs(root)!=-1;
}


Q)LCA
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T data;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T data) {
            this->data = data;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/
TreeNode<int>* util(TreeNode<int> *root, int x, int y){
    if(!root){
        return NULL;
    }
    
    //we found one of the required nodes 
    if(root->data==x || root->data==y){
        return root;
    }
    
    //to check which nodes have been found
    //in the subtree of current node
    TreeNode<int>* l=util(root->left,x,y);
    TreeNode<int>* r=util(root->right,x,y);
    
    //if we found both node in the subtree
    //of current node it means current node
    //is LCA of required nodes
    if(l && r){
        return root;
    }
    
    //if we found only one of the nodes in 
    //subtree of current node
    // we return the node we found
    //in subtree of current node
    //it indicates that lca lies somewhere above
    //the current node and that one of the required values
    //is found in subtree of upper nodes.
    //or it means that one node lies in subtree of
    //another node
    if(l and !r){
        return l;
    }else{
        return r;
    }
}
int lowestCommonAncestor(TreeNode<int> *root, int x, int y)
{
	//    Write your code here
   TreeNode<int>* ans=util(root,x,y);
    return ans->data;
}

------------------------------------------------------------------------------------------------
Day 19: Binary Tree part-III
Q)Maximum Path Sum
int util(TreeNode* root,int &ans){
	if(!root){
		return 0;
	}
	
	//if left or right is negative
	//we ignore it
	int left=util(root->left,ans);
	int right=util(root->right,ans);
	
	int straightPath=max({root->val,left+root->val,right+root->val});
	int curvedPath=left+right+A->val;
	
	ans=max({ans,curvedPath,straightPath});
	
	return straightPath;
}
int maxPathSum(TreeNode* root) {
	//a path in a tree
	//is simply a path from one node to other
	//in which we dont visit the same node twice.
	int ans=INT_MIN;
	util(root,ans);
	return ans;
}

Q)Construct binary Tree from inorder and preorder Traversal
TreeNode* util(vector<int> &inorder,vector<int> &preorder,unordered_map<int,int> &mp,int start,int end,int &idx){
	if(start>end) return NULL;
	
	//assigning value of current element to root
	TreeNode* root=new TreeNode(preorder[idx]);
	//moving pointer to next element
	idx++;
	
	//when we are on leaf node 
	if(start==end){
		return root;
	}
	//the mid value is index of current root
	int mid=m[root->val];
	
	//in preorder traversal we are iterating from
	//start and since preorder traversal is like NLR
	//so after finding the node we build left subtree
	//then we build right subtree.
	root->left=util(inorder,preorder,mp,start,mid-1,idx);
	root->right=util(inorder,preorder,mp,mid+1,end,idx);
	
	return root;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
   unordered_map<int,int> mp;
   
   for(int i=0;i<size(inorder);i++){
		mp[inorder[i]]=i;
   }
   // int idxOfRootInInorder=mp[preorder[0]];
	 int n=preorder.size()
	TreeNode* ans=util(inorder,preorder,mp,0,n-1,0);
	
	return ans;
}

Q) Construct Binary Tree from Inorder and Postorder Traversal
TreeNode* util(vector<int> &inorder,vector<int> &postorder,unordered_map<int,int> &mp,int start,int end,int &idx){
		if(start>end){
				return NULL;
		}
		
		TreeNode* root=new TreeNode(postorder[idx]);
		idx-=1;
		//leaf node
		if(start==end){
				return root;
		}
		
		int mid=mp[root->val];
		//in preorder traversal we need to construct right first
		//since we are iterating postorder in reverse and
		//postorder in LRN so its reverse woulf be NRL 
		//i.e after a node comes it's right subtree and then comes it's
		//left subtree so we need to construct right subtree first
		root->right=util(inorder,postorder,mp,mid+1,end,idx);       
			root->left=util(inorder,postorder,mp,start,mid-1,idx);
		
		return root;
}
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
		unordered_map<int,int> mp;
		int n=inorder.size();
		for(int i=0;i<n;i++){
				mp[inorder[i]]=i;
		}
		
		int idx=n-1;
		TreeNode* ans=util(inorder,postorder,mp,0,n-1,idx);
		return ans;
}


q)flatten a linkdd list
approach 1))O(n^2)
void flatten(TreeNode* root) {
	if(!root){
			return;
	}
	
	//save left and right child
	//of current root in temporary variables
	TreeNode* tmpLeft=root->left;
	TreeNode* tmpRight=root->right;
	
	//then remove left child
	root->left=NULL;
	
	//repeat the above 2 steps for both left and right childs
	flatten(tmpLeft);
	flatten(tmpRight);
	
	//now attach the removed left child as right child of current root
	//as given in question
	root->right=tmpLeft;
	
	//make a temp variable which points to current root
	TreeNode* curr=root;
	//now keep moving downwards in the right side of tree until last right node
	while(curr->right)curr=curr->right;
	//attach the right child of current node there
	curr->right=tmpRight;
}

//method 2(Morris traversal)
------------------------------------------------------------------------------------------------
Day 20: Binary Search Tree
Q)BST from preorder
TreeNode* util(vector<int> &preorder,int lo,int hi,int &idx){
		if(idx>=preorder.size()) return NULL;
		
		if(preorder[idx]<lo || preorder[idx]>hi){
				return NULL;
		}
		
		TreeNode* root=new TreeNode(preorder[idx]);
		idx+=1;
		root->left=util(preorder,lo,root->val,idx);
		root->right=util(preorder,root->val,hi,idx);
		
		return root;
}
TreeNode* bstFromPreorder(vector<int>& preorder) {
		int lo=INT_MIN;
		int hi=INT_MAX;
		int idx=0;
		TreeNode* ans=util(preorder,lo,hi,idx);
		
		return ans;
}


Q)Find lca of 2 nodes in bst
 TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // TreeNode* ans=root;
        
        while(root){
					//if current node is smaller than both given node
					//we need to find a bigger value node so we
					//move to right subtree and move to left subtree for
					//vice versa
            if(q->val>root->val && p->val>root->val){
                root=root->right;
            }else if(q->val<root->val && p->val<root->val){
                root=root->left;
            }else{
								//if root lies in the range of the 2 given nodes
								//we return it
                return root;
            }
        }
        
        return NULL;
    }
    
Q)find inorder predecessor and successor in a bst
void findPreSuc(Node* root, Node*& pre, Node*& suc, int key)
{

// Your code goes here
		if(!root){
			return;
		}
		
		if(root->data==key && root->right && root->left){
			if(root->left){
				Node* tmp=root;
			tmp=tmp->left;
			while(tmp){
				tmp=tmp->right;
			}
			pre=tmp;
			}
			
			if(root->right){
			Node* tmp1=root;
			tmp1=tmp1->right;
			while(tmp1){
				tmp1=tmp1->left;
			}
			suc=tmp1;
			}
			return;
		}
		
		if(root->data>key){
			suc=root->data;
			findPreSuc(root->left,pre,suc,key);
		}else if(root->data<key){
			pre=root->data;
			root->data=findPreSuc(root->right,pre,suc,key);
		}
}

Q) ceil in bst(same logic works for floor)
void util(Node* root,int &v,int input){
    if(!root){
        
        return;
    }
    
    if(input==root->data){
        v=root->data;
        return;
    }
    //if current root value is smaller than input
    //we move to right subtree
    if(input>root->data){
        util(root->right,v,input);
    }else{
				//if current root value is greater than input 
				//then it is a possible candidate for being a ceil value
				//of given input
				
        v=root->data;
        util(root->left,v,input);
    }
    
   
}

int findCeil(struct Node* root, int input) {
    // your code here
    int v=0;
    util(root,v,input);
    
    // int idx=lower_bound(begin(v),end(v),input)-begin(v);
    
    return v;
}

Q) BST iterator
//my approach
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class BSTIterator {
public:
    vector<int> v;
    int key=INT_MIN;
    void inorder(TreeNode* root){
        if(!root){
            return;
        }
        
        inorder(root->left);
        v.push_back(root->val);
        inorder(root->right);
    }
    BSTIterator(TreeNode* root) {
        inorder(root);
    }
    
    int next() {
        auto it=upper_bound(begin(v),end(v),key);
        if(it!=v.end()){
            key=*it;
        }
        return *it;
    }
    
    bool hasNext() {
        auto it=upper_bound(begin(v),end(v),key);
        return it==end(v)?0:1;
        
    }
};

//stack solution
class BSTIterator {
stack<Node*> st;
public:
    void pushAll(TreeNode* root){
			//will make sure smallest element is at top
			while(root!=NULL){
				st.push(root);
				root=root->left;
			}
    }
    BSTIterator(TreeNode* root) {
				pushAll(root);
    }
    //after finding smallest element greater than 
    //current element we push all elements just greater
    //than the current element into the stack
    int next() {
        TreeNode* nxt=st.top();
        st.pop();
        pushAll(nxt->right);
        return nxt->val;
    }
    
    bool hasNext() {
      return !st.empty();
        
    }
};
/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
 
 Q)Serialize and deserialize a tree
 /**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
public:
    void serialize(TreeNode* root,string &s){
        if(!root){
            s+="null,";
            return;
        }
        
        s+=to_string(root->val)+",";
        serialize(root->left,s);
        serialize(root->right,s);
    }
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string s="";
        serialize(root,s);
        return s;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(vector<string> &v,int &idx){
        if(idx>=v.size()||v[idx]=="null"){
            idx++;
            return NULL;
        }
        
        TreeNode* root=new TreeNode(stoi(v[idx++]));
        root->left=deserialize(v,idx);
        root->right=deserialize(v,idx);
        
        return root;
    }
    TreeNode* deserialize(string data) {
        vector<string> v;
        
        int start=0;
        int end=data.find(",");
        
        while(end!=-1){
            v.push_back(data.substr(start,end-start));
            start=end+1;
            end=data.find(",",start);
        }
        int idx=0;
        TreeNode* ans=deserialize(v,idx);
        return ans;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));

Q)largest BST in binary Tree

------------------------------------------------------------------------------------------------
Day 21: Binary Search Tree Part-II
------------------------------------------------------------------------------------------------
Day 22: Binary Trees[Miscellaneous]
------------------------------------------------------------------------------------------------
Day 23: Graph
Q)Dijkstra Algo(Network delay time)
class Solution {
public:
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
	vector<vector<pair<int,int>>> adj(105);
	
	
	for(int i=0;i<times.size();i++){
			adj[times[i][0]].push_back({times[i][1],times[i][2]});
	}
	
	//making min heap
	
	priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>  pq;
	
	
	vector<int> dist(105,INT_MAX);
	dist[k]=0;
	pq.push({0,k});
	vector<int> vis(105,0);
	
	while(!pq.empty()){
			int node=pq.top().second;;
			pq.pop();
			if(!vis[node]){
					
			vis[node]=1;
			for(auto it:adj[node]){
					int curr_child=it.first;
					int edge_curr_child=it.second;
					
					if(dist[curr_child]>dist[node]+edge_curr_child){
							dist[curr_child]=dist[node]+edge_curr_child;
							pq.push({dist[curr_child],curr_child});
					}
			}
			}
	}
	
	int ans=INT_MIN;
	for(int i=1;i<=n;i++){
			if(i!=k && dist[i]==INT_MAX) return -1;
			ans=max(ans,dist[i]);
	}
	
	return ans;
}
};

Q)All pair Shortest Path algo(floyd warshall algo)
int floydWarshallAlgo(vector<vector<pair<int,int>> adj){
	vector<vector<int>> dist(100005,vector<int>(100005,-1));
	
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			if(i==j){
				dist[i][j]=0;
			}else{
				dist[i][j]=INT_MAX;
			}
		}
	}
	
	int n,m;
	cin>>n>>m;
	for(int i=0;i<m;++i){
		int x,y,wt;
		cin>>x>>y>>wt;
		//this means when we have allowed
		//calculation of path using 0 nodes 
		dist[x][y]=wt;
	}
	
	//calculating all pair shoertest path
	for(int k=1;k<=n;k++){
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				//if any of them is infinity it simply means
				//path doesnt exist between i and k or k and j or both
				if(dist[i][k]!=INT_MAX && dist[k][j]!=INT_MAX)
				dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
			}
		}
	}
}
------------------------------------------------------------------------------------------------
Day 24: Graph Part-II
------------------------------------------------------------------------------------------------
Day 25: Dynamic Programming
------------------------------------------------------------------------------------------------
Day 26: Dynamic Programming Part-II
------------------------------------------------------------------------------------------------
Day 27: Trie
Q1)Implement Trie
/*
    Your Trie object will be instantiated and called as such:
    Trie* obj = new Trie();
    obj->insert(word);
    bool check2 = obj->search(word);
    bool check3 = obj->startsWith(prefix);
 */

struct Node{
    Node* links[26];
    bool flag=false;
    void put(char ch,Node* node){
        links[ch-'a']=node;
    }
    bool containsKey(char ch){
        return links[ch-'a']!=NULL;
    }
    
    Node* get(char ch){
        return links[ch-'a'];
    }
    
    void setEnd(){
        flag=true;
    }
}
class Trie {
private:
    Node* root;
public:

    /** Initialize your data structure here. */
    Trie() {
        root=new Node();
    }

    /** Inserts a word into the trie. */
    void insert(string word) {
        Node* node=root;
        
        for(int i=0;i<word.size();i++){
            if(!node->containsKey(word[i])){
                node->put(word[i],new Node());
            }
            node=node->get(word[i]);
        }
        
        node->setEnd();
    }

    /** Returns if the word is in the trie. */
    bool search(string word) {
        Node* node=root;
        
        for(int i=0;i<word.size();i++){
            if(!node->containsKey(word[i])){
                return 0;
            }
            node=node->get(word[i]);
        }
        
        if(node->flag==1){
            return true;
        }
        
        return false;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Node* node=root;
        
        for(int i=0;i<prefix.size();i++){
            if(!node->containsKey(prefix[i])){
                return false;
            }
            
            node=node->get(prefix[i]);
        }
        return 1;
    }
};

Q2)Implement trie 2

struct Node {
  Node * links[26];
  int cntEndWith = 0;
  int cntPrefix = 0;

  bool containsKey(char ch) {
    return (links[ch - 'a'] != NULL);
  }
  Node * get(char ch) {
    return links[ch - 'a'];
  }
  void put(char ch, Node * node) {
    links[ch - 'a'] = node;
  }
  void increaseEnd() {
    cntEndWith++;
  }
  void increasePrefix() {
    cntPrefix++;
  }
  void deleteEnd() {
    cntEndWith--;
  }
  void reducePrefix() {
    cntPrefix--;
  }
  int getEnd() {
    return cntEndWith;
  }
  int getPrefix() {
    return cntPrefix;
  }
};
class Trie {
  private:
    Node * root;

  public:
    /** Initialize your data structure here. */
    Trie() {
      root = new Node();
    }

  /** Inserts a word into the trie. */
  void insert(string word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (!node -> containsKey(word[i])) {
        node -> put(word[i], new Node());
      }
      node = node -> get(word[i]);
      node -> increasePrefix();
    }
    node -> increaseEnd();
  }

 int countWordsEqualTo(string &word)
    {
        Node *node = root;
        for (int i = 0; i < word.length(); i++)
        {
            if (node->containsKey(word[i]))
            {
                node = node->get(word[i]);
            }
            else
            {
                return 0;
            }
        }
        return node->getEnd();
    }


  int countWordsStartingWith(string & word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (node -> containsKey(word[i])) {
        node = node -> get(word[i]);
      } else {
        return 0;
      }
    }
    return node -> getPrefix();
  }

  void erase(string & word) {
    Node * node = root;
    for (int i = 0; i < word.length(); i++) {
      if (node -> containsKey(word[i])) {
        Pnode = node -> get(word[i]);
        node -> reducePrefix();
      } else {
        return;
      }
    }
    node -> deleteEnd();
  }
};

Q3)Complete String
struct Node{
	Node* links[26];
	bool flag=0;
	
	void put(char ch,Node* node){
		links[ch-'a']=node;
	}
	
	Node* get(char ch){
		return links[ch-'a'];
	}
	
	bool containsKey(char ch){
		return links[ch-'a']!=NULL;
	}
	
	void setEnd(){
		flag=1;
	}
};

class Trie{
	private:
		Node* root;
	public:
		Trie(){
			root=new Node();
		}
		
		void insert(string word){
			Node* node=root;
			for(int i=0;i<word.size();i++){
				if(!node->containsKey(word[i])){
					node->put(word[i],new Node());
				}
				node=node->get(word[i]); 
			}
			node->setEnd();
		}
		
		bool search(string word){
			Node* node=root;
			for(int i=0;i<word.size();i++){
				if(!node->containsKey(word[i])){
					return false;
				}
				node=node->get(word[i]); 
			}
			if(node->flag==1){
				return 1;
			}
			
			return 0;
		}
		
		bool checkIfPrefixExists(string word){
			bool ans=true;
			Node* node=root;
			for(int i=0;i<word.length();i++){
				if(node->containsKey(word[i])){
					node=node->get(word[i]);
					if(node->flag==0) return false;
				}
				return false;
			}
			
			return true;
		}
};
string completeString(int n, vector<string> &a){
    // Write your code here.
    
    Trie trie;
    for(auto &it:a){
			trie.insert(it)
    }
    
    string ans="";
    for(auto &it:a){
			if(trie.checkIfPrefixExists(it)){
				if(it.length()>ans.length()){
					ans=it;
				}else if(it.length()==ans.length() && it<ans){
					ans=it;
				}
			}
    }
    
    if(ans==""){
			return "None"
    }
    
    return ans;
}
//map approach
#include<bits/stdc++.h>
string completeString(int n, vector<string> &a){
   
    unordered_map<string,int> mp;
    
    for(auto it:a){
        mp[it]=1;
    }
    string ans="";
    for(int i=0;i<a.size();i++){
        int flag=0;
        for(int j=0;j<a[i].size();j++){
            string t=a[i].substr(0,j+1);
            if(mp.find(t)==mp.end()){
                flag=1;
                break;
            }
        }
        if(!flag){
                if(a[i].size()>ans.size()){
                    ans=a[i];
                }else if(a[i].size()==ans.size() && a[i]<ans){
                  ans=a[i];  
                }
            }
    }
    
    if(ans==""){
        return "None";
    }
    
    return ans;
}

Q3)Count distinct substrings:
struct Node{
	Node* links[26];
	bool flag=0;
	
	void put(char ch,Node* node){
		links[ch-'a']=node;
	}
	
	Node* get(char ch){
		return links[ch-'a'];
	}
	
	bool containsKey(char ch){
		return links[ch-'a']!=NULL;
	}
	
	void setEnd(){
		flag=1;
	}
	
	bool isEnd(){
		return flag;
	}
};
int countDistinctSubstrings(string &s)
{
    //    Write your code here.
    Node* root=new Node();
    int ans=0;
    for(int i=0;i<s.size();i++){
			string t=s.substr(i);
			Node* node=root;
			for(int j=0;j<t.size();j++){
				if(!node->containsKey(t[j])){
					ans++;
					node->put(t[j],new Node());
				}
				node=node->get(t[j]);
			}
    }
    
    return ans+1;
}

Q4)Power set
vector<vector<int>> pwset(vector<int>v)
{
    //Write your code here
    vector<vector<int>> ans;
    
    int n=v.size();
    int s=pow(2,n);
    
    for(int i=0;i<s;i++){
        vector<int> tmp;
        for(int j=0;j<32;j++){
            if(i&(1<<j)){
                tmp.push_back(v[j]);
            }
        }
        ans.push_back(tmp);
    }
    
    return ans;
}

number of 1s are even then xor=0
number of 1s are odd then xor=1 

check if ith bit is set or not for x:
	if x&(1<<i) then bit is set
	else bit is not set

turn on ith bit of x:
	x=x or (1<<i)


Q5)Maximum xor
struct Node{
	Node* links[2];
	
	void put(int bit,Node* node){
		links[bit]=node;
	}
	
	Node* get(int bit){
		return links[bit];
	}
	
	bool containsKey(int bit){
		return links[bit]!=NULL;
	}
	
};

class Trie{
	private:
		Node* root;
	public:
		Trie(){
			root=new Node();
		}
		
		void insert(int num){
			Node* node=root;
			for(int i=31;i>=0;i--){
				 int bit=num&(1<<i);
				 if(!node->containsKey(bit)){
					node->put(bit,new Node());
				 }
				 node=node->get(bit);
			}
		}
		
		int getMax(int num){
			Node* node=root;
			int maxNum=0;
			for(int i=31;i>=0;i--){
				int bit=num&(1<<i);
				if(!node.containsKey(1-bit)){
					maxNum=maxNum | (1<<i);
					node=node->get(1-bit);
				}else{
					node=node->get(bit);
				}
			}
			
			return maxNum;
		}
};

int maxXOR(int n, int m, vector<int> &arr1, vector<int> &arr2) 
{
    // Write your code here. 
   Trie trie;
   
   for(auto it:arr1){
		trie.insert(it);
   }
   int maxi=0;
   for(auto it:arr2){
		maxi=max(maxi,trie.getMax(it));
   }
   
   return maxi;
      
}


------------------------------------------------------------------------------------------------
Day 28: Operating System Revision (Refer Sheet for OS Questions) 
------------------------------------------------------------------------------------------------
Day 29: DBMS Revision (Refer Sheet for DBMS Questions) 
------------------------------------------------------------------------------------------------
Day 30: Computer Networks Revision (Refer Sheet for CN Questions)  
------------------------------------------------------------------------------------------------
Day 31: Project Overview

