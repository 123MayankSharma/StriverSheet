#include<bits/stdc++.h>
using namespace std;

map in c++

struct SegmentTreeNode{
  int start,end;
  SegmentTreeNode *left,*right;
  int sum;
  
  SegmentTreeNode(int start,int end){
    start=start;
    end=end;
    left=NULL;
    right=NULL;
    sum=0;
  }
}



class NumArray {
SegmentTreeNode* root=NULL;
public:
    SegmentTreeNode* buildTree(vector<int> &nums,int start,int end){
      if(start>end){
        return NULL;
      }
      
      SegmentTreeNode* root=new SegmentTreeNode(start,end);
      if(start==end){
        root->sum=nums[start];
      }else{
        int mid=start+(end-start)/2;
        root->left=buildTree(nums,start,mid);
        root->right=buildTree(nums,mid+1,end);
        root->sum=root->left->sum + root->right->sum;
      }
      
      return root;
    }
    NumArray(vector<int>& nums) {
        root=buildTree(nums,0,nums.size()-1);
    }
    //o(log(n))
    void updateUtil(SegmentTreeNode* root,int index,int val){
      if(root->start==root->end){
        root->sum=val;
      }else{
        //parent nodes across the path
        int mid=root->start+(root->end-root->start)/2;
        if(index<=mid){
          updateHelper(root->left,index,val);
        }else{
          updateHelper(root->right,index,val);
        }
        root->sum=root->left->sum+root->right->sum;
      }
    }
    void update(int index, int val) {
        updateUtil(root,index,val);
    }
    
    int sumRangeUtil(SegmentTreeNode* root,int left,int right){
      //if oyu found out range that is given
      //we return it
      if(root->end==end && root->start==start){
        return root->sum;
      }
      
      int mid=root->start+(root->end-root->start)/2;
      if(end<=mid){
        return sumRangeUtil(root->left,start,end);
      }else if(start>=mid+1){
        return sumRangeUtil(root->right,start,end);
      }else{
        return sumRangeUtil(root->left,start,mid)+sumRangeUtil(root->right,mid+1,end);
      }
    }
    int sumRange(int left, int right) {
     return sumRangeUtil(root,left,right);   
    }
};
