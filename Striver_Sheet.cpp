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
    vector<vector<int>> ans;
		
}
	

------------------------------------------------------------------------------------------------
Day 8: Greedy Algorithm
------------------------------------------------------------------------------------------------
Day 9: Recursion
------------------------------------------------------------------------------------------------
Day 10: Recursion and Backtracking
------------------------------------------------------------------------------------------------
Day 11: Binary Search
------------------------------------------------------------------------------------------------
Day 12: Heaps
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
------------------------------------------------------------------------------------------------
Day 18: Binary Tree part-II
------------------------------------------------------------------------------------------------
Day 19: Binary Tree part-III
------------------------------------------------------------------------------------------------
Day 20: Binary Search Tree
------------------------------------------------------------------------------------------------
Day 21: Binary Search Tree Part-II
------------------------------------------------------------------------------------------------
Day 22: Binary Trees[Miscellaneous]
------------------------------------------------------------------------------------------------
Day 23: Graph
------------------------------------------------------------------------------------------------
Day 24: Graph Part-II
------------------------------------------------------------------------------------------------
Day 25: Dynamic Programming
------------------------------------------------------------------------------------------------
Day 26: Dynamic Programming Part-II
------------------------------------------------------------------------------------------------
Day 27: Trie
------------------------------------------------------------------------------------------------
Day 28: Operating System Revision (Refer Sheet for OS Questions) 
------------------------------------------------------------------------------------------------
Day 29: DBMS Revision (Refer Sheet for DBMS Questions) 
------------------------------------------------------------------------------------------------
Day 30: Computer Networks Revision (Refer Sheet for CN Questions)  
------------------------------------------------------------------------------------------------
Day 31: Project Overview

