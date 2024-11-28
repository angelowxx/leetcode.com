/**
 * @Author: Wang Xinxiang
 * @Description: A  TreeNode Class
 * @DateTime: 7/28/2024 11:04 AM
 */

public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
