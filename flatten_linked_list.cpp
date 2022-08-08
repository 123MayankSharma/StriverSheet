karan_8082's avatar
karan_8082
1761

Last Edit: July 27, 2022 7:41 AM

4.0K VIEWS

UPVOTE IF HELPFuuL

APPROACH -> O ( N ) SPACE
By using brute force, we can traverse the tree in preorder manner and store the result in a vector.
Later using that vector , we can arrange nodes, such as
vector[i] -> right = vector[i+1];
vector[i] -> left = NULL;

APPROACH -> O ( 1 ) SPACE

From the diagram given , it can be seen all nodes are present on the right.
All the nodes in left subtree come before the nodes in right subtree.

For each node i

IF there is no left node -> move to next right node.
IF LEFT IS PRESENT ->
  Store the right subtree
  ADD left subtree to right of root,
  Now add the stored right subtree to the rightmost node of current tree.
Also make node -> left =NULL.


//see below iimage for bettr clarity


https://assets.leetcode.com/users/images/e6d4b1fa-b7ec-4056-a6ba-27a8fb5ea60b_1658887502.7798896.png


//code
  TreeNode* rightmost(TreeNode* root){
        if (root->right==NULL) return root;
        return rightmost(root->right);
    }
    
    void flatten(TreeNode* root) {
        if (root==NULL) return;
        TreeNode* nextright;
        TreeNode* rightMOST;
        
        while (root){
            
            if (root->left){
                rightMOST = rightmost(root->left);
                nextright = root->right;
                root->right = root->left;
                root->left=NULL;
                rightMOST->right=nextright;
            }
            root=root->right;
        }
    }
};
