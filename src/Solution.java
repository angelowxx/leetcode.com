import java.util.*;

/**
 * @Author: Wang Xinxiang
 * @Description: This is a class containing leetcode problems answers
 * @DateTime: 12/11/2023 11:26 AM
 */



public class Solution {


    public int countPalindromes(List<String> arr) {
        // Write your code here
        Map<String, Integer> map = new HashMap<>();
        int[] cnt = new int[26];
        for(int i=0; i<arr.size(); i++){
            getOperationTimes(arr.get(i), cnt, map);
        }
        String[] arrs = new String[arr.size()];
        int i=0;
        for(String ar : map.keySet()){
            arrs[i++] = ar;
        }
        Arrays.sort(arrs, (a,b)->map.get(a)-map.get(b));
        int res = 0;
        for(String ar:arrs){
            System.out.println(ar+","+map.get(ar));
            if(map.get(ar)==0){
                res++;
            }
            else{
                char[] ars = ar.toCharArray();
                int[] cntCur = cnt.clone();
                boolean sign = true;
                for(int l=0, r=ars.length-1; l<r; l++,r--){
                    char left = ars[l];
                    char right = ars[r];
                    if(left!=right){
                        if(cntCur[left-'a']>cntCur[right-'a']){
                            cntCur[left-'a'] -= 2;
                            if(cntCur[left-'a']<0){
                                sign = false;
                                break;
                            }
                        }
                        else{
                            cntCur[right-'a'] -= 2;
                            if(cntCur[right-'a']<0){
                                sign = false;
                                break;
                            }
                        }
                    }
                }
                if(sign){
                    cnt = cntCur;
                    res++;
                }
            }
        }
        return res;
    }

    public void getOperationTimes(String arr, int[] cnt, Map<String, Integer> cntMap){
        int res = 0;
        char[] arrs = arr.toCharArray();
        int l = 0, r = arrs.length-1;
        List<Integer> list = new ArrayList<>();
        while(l<r){
            if(arrs[l]!=arrs[r]){
                cnt[arrs[l]-'a']++;
                cnt[arrs[r]-'a']++;
                res += 2;
                list.add(l);
                list.add(r);
            }
            l++;
            r--;
        }
        if(l==r){
            cnt[arrs[r]-'a']++;
        }
        for(int i=0; i<list.size(); i++){
            int ri = arrs.length-i-1;
            for(int j=0; j<list.size(); j++){
                if(j!=ri&&arrs[ri]==arrs[j]){
                    res -= 2;
                    cnt[arrs[ri]-'a']--;
                    cnt[arrs[j]-'a']--;
                    list.remove(j);
                    swap(arrs, i, j);
                    break;
                }
            }
        }
        arr = String.valueOf(arrs);
        cntMap.put(arr, res);
    }

    public void swap(char[] arr, int i, int j){
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    public long countSimilarPairs(List<String> words) {
        // Write your code here
        Map<String, Long> map = new HashMap<>();
        for(String word : words){
            String w = getLetters(word);
            map.put(w, map.getOrDefault(w, 0l)+1);
        }
        long res = 0;
        for(String letters : map.keySet()){
            long cur = map.get(letters);
            res += cur*(cur-1)/2;
        }
        return res;
    }

    public long countNumbers(long n){
        n--;
        long res = n;
        n--;
        while(n>0){
            res *= n;
            n--;
        }
        return res;
    }
    public String getLetters(String word){
        Set<Character> set = new HashSet<>();
        for(char w : word.toCharArray()){
            set.add(w);
        }
        char[] res = new char[set.size()];
        int i=0;
        for(char l:set){
            res[i++] = l;
        }
        Arrays.sort(res);
        return String.valueOf(res);
    }

    public String getSpreadsheetNotation(long n) {
        long l = n/702;
        long r = n%702;
        if(r!=0){
            l++;
        }
        if(r==0){
            r=702;
        }
        return l+convertNumberToString((int)r);
    }

    public String convertNumberToString(int n){
        if(n<0){
            return "";
        }
        int l = n/26;
        int r = n%26;
        if(r>0){
            l++;
        }
        if(r==0){
            r=26;
        }
        l--;
        char[] res = new char[2];
        res[0] = (char)('A'+l-1);
        res[1] = (char)('A'+r-1);
        if(l==0){
            return String.valueOf(res[1]);
        }
        else{
            return String.valueOf(res);
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 24. 两两交换链表中的节点
    * @DateTime: 11/20/2024 12:36 AM
    * @Params: 
    * @Return 
    */
    public ListNode swapPairs2(ListNode head) {
        ListNode preHead = new ListNode();
        preHead.next = head;
        ListNode pre = preHead;
        while(head!=null&&head.next!=null){
            swapTwoNodes(head, pre);
            pre = head;
            head = head.next;
        }
        return preHead.next;
    }

    public void swapTwoNodes(ListNode head, ListNode pre){
        pre.next = head.next;
        head.next = pre.next.next;
        pre.next.next = head;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 1862. 向下取整数对和
    * @DateTime: 11/14/2024 1:06 PM
    * @Params: 
    * @Return 
    */
    public int sumOfFlooredPairs(int[] nums) {
        int mod = 1000000007;
        int max = Integer.MIN_VALUE;
        for(int n:nums){
            max = Math.max(max, n);
        }
        int[] cnt = new int[max+1];
        for (int num: nums) {
            ++cnt[num];
        }
        // 预处理前缀和
        int[] pre = new int[max+1];
        for (int i = 1; i <= max; ++i) {
            pre[i] = pre[i - 1] + cnt[i];
        }

        long ans = 0;
        for (int y = 1; y <= max; ++y) {
            // 一个小优化，如果数组中没有 y 可以跳过
            if (cnt[y]>0) {
                for (int d = 1; d * y <= max; ++d) {
                    ans += cnt[y]*d*(pre[Math.min((d + 1) * y - 1, max)] - pre[d * y - 1]);
                    ans = ans%mod;
                }
            }
        }
        return (int)ans;

    }

    public int search(int[] nums, int target, int l, int r){
        if(l>r){
            return r;
        }
        int mid = (l+r)/2;
        if(nums[mid]==target){
            return mid;
        }
        else if(nums[mid]<target){
            return search(nums, target, mid+1, r);
        }
        else{
            return search(nums, target, l, mid-1);
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 160. 相交链表
    * @DateTime: 11/13/2024 4:00 PM
    * @Params: 
    * @Return 
    */
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        int l1 = getLength(headA);
        int l2 = getLength(headB);
        if(l1>l2){
            for(int i=0; i<l1-l2; i++){
                headA = headA.next;
            }
        }
        else{
            for(int i=0; i<l2-l1; i++){
                headB = headB.next;
            }
        }
        while(headA!=headB){
            headA = headA.next;
            headB = headB.next;
        }
        return headA;

    }

    public int getLength(ListNode head){
        int res = 0;
        while(head!=null){
            res++;
            head = head.next;
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 240. 搜索二维矩阵 II
    * @DateTime: 11/13/2024 3:10 PM
    * @Params: 
    * @Return 
    */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m=0,n=matrix[0].length-1;
        while(m<matrix.length&&n>=0){
            if(matrix[m][n]==target){
                return true;
            }
            else if(matrix[m][n]>target){
                n--;
            }
            else{
                m++;
            }
        }
        return false;
    }

    public boolean checkMatrix(int[][] matrix, int target, int m, int n){
        if(m>=matrix.length||n<0){
            return false;
        }
        if(matrix[m][n]==target){
            return true;
        }
        else if(matrix[m][n]<target){
            return checkMatrix(matrix, target, m+1, n);
        }
        else{
            return checkMatrix(matrix, target, m, n-1);
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 48. 旋转图像
    * @DateTime: 11/12/2024 10:41 PM
    * @Params: 
    * @Return 
    */
    public void rotate2(int[][] matrix) {
        if(matrix==null||matrix.length<2||matrix.length!=matrix[0].length){
            return;
        }
        for(int i=0; i<matrix.length; i++){
            for(int j=i; j<matrix.length-i-1; j++){
                rotateOne(matrix, i, j);
            }
        }
    }

    public void rotateOne(int[][] matrix, int m, int n){
        swap2D(matrix, m, n, n, matrix.length-1-m);
        swap2D(matrix, m, n, matrix.length-1-m, matrix.length-1-n);
        swap2D(matrix, m, n,matrix.length-1-n, m);
    }
    public void swap2D(int[][] matrix, int m1, int n1, int m2, int n2){
        int tmp = matrix[m1][n1];
        matrix[m1][n1] = matrix[m2][n2];
        matrix[m2][n2] = tmp;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 54. 螺旋矩阵
    * @DateTime: 11/12/2024 10:21 PM
    * @Params: 
    * @Return 
    */
    public List<Integer> spiralOrder2(int[][] matrix) {
        if(matrix==null||matrix.length==0||matrix[0].length==0){
            return new ArrayList<Integer>();
        }
        List<Integer> res = new ArrayList<>();
        int[][] directions = new int[][]{{0,1}, {1,0}, {0,-1}, {-1,0}};
        add2List(matrix, 0, 0, directions, 0, res);
        return res;
    }

    public void add2List(int[][] matrix, int m, int n, int[][] directions, int d, List<Integer> list){
        list.add(matrix[m][n]);
        matrix[m][n] = 101;
        for(int i=0; i<4; i++, d=(d+1)%4){
            int nextM = m + directions[d][0];
            int nextn = n + directions[d][1];
            if(nextM>=0&&nextM<matrix.length&&nextn>=0&&nextn<matrix[0].length&&matrix[nextM][nextn]<=100){
                add2List(matrix, nextM, nextn, directions, d, list);
                return;
            }
        }
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 73. 矩阵置零
    * @DateTime: 11/12/2024 9:38 PM
    * @Params: 
    * @Return 
    */
    public void setZeroes2(int[][] matrix) {
        boolean[] sign  = new boolean[1];
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                if(matrix[i][j]==0){
                    setZero(matrix, i, j, sign);
                }
            }
        }
        for(int i=1; i<matrix.length; i++){
            for(int j=1; j<matrix[0].length; j++){
                if(matrix[i][0]==0||matrix[0][j]==0){
                    matrix[i][j] = 0;
                }
            }
        }
        if(matrix[0][0]==0){
            for(int i=0; i<matrix[0].length; i++){
                matrix[0][i] = 0;
            }
        }
        if(sign[0]){
            for(int i=0; i<matrix.length; i++){
                matrix[i][0] = 0;
            }
        }
    }
    public void setZero(int[][] matrix, int m, int n, boolean[] sign){
        matrix[m][0] = 0;
        if(n!=0){
            matrix[0][n] = 0;
        }
        else{
            sign[0] = true;
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 41. 缺失的第一个正数
    * @DateTime: 11/11/2024 11:07 PM
    * @Params: 
    * @Return 
    */
    public int firstMissingPositive(int[] nums) {
        for(int i=0; i<nums.length; i++){
            while(nums[i]>0&&nums[i]<=nums.length&&nums[i]!=i+1&&nums[nums[i]-1]!=nums[i]){
                swap2(nums, i, nums[i]-1);
            }
        }
        int res = 0;
        for(; res<nums.length; res++){
            if(nums[res]!=res+1){
                break;
            }
        }
        return res+1;
    }

    public void swap2(int[] nums, int l, int r){
        int tmp = nums[l];
        nums[l] = nums[r];
        nums[r] = tmp;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 238. 除自身以外数组的乘积
    * @DateTime: 11/11/2024 10:57 PM
    * @Params: 
    * @Return 
    */
    public int[] productExceptSelf2(int[] nums) {
        int[] dp = new int[nums.length];
        dp[nums.length-1] = 1;
        for(int i=nums.length-2; i>=0; i--){
            dp[i] = dp[i+1]*nums[i+1];
        }
        int[] res = new int[nums.length];
        int mul = 1;
        for(int i=0; i<nums.length; i++){
            res[i] = dp[i]*mul;
            mul *= nums[i];
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 189. 轮转数组
    * @DateTime: 11/11/2024 10:48 PM
    * @Params: 
    * @Return 
    */
    public void rotate2(int[] nums, int k) {
        k = k%nums.length;
        reverse2(nums, 0, nums.length-1);
        reverse2(nums, 0, k-1);
        reverse2(nums,k, nums.length-1);
    }

    public void reverse2(int[] nums, int l, int r){
        while(l<r){
            swap2(nums, l++, r--);
        }
    }



    /**
    * @Author: Wang Xinxiang
    * @Description: 56. 合并区间
    * @DateTime: 11/11/2024 1:08 AM
    * @Params: 
    * @Return 
    */
    public int[][] merge(int[][] intervals) {
        List<int[]> q = new ArrayList<>();
        Arrays.sort(intervals, (a,b)->a[0]-b[0]);
        int[] cur = intervals[0];
        for(int[] interval : intervals){
            if(cur[1]>=interval[0]){
                cur[1] = Math.max(cur[1], interval[1]);
            }
            else{
                q.add(cur);
                cur = interval;
            }
        }
        q.add(cur);
        return q.toArray(new int[0][]);
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 53. 最大子数组和
    * @DateTime: 11/11/2024 12:55 AM
    * @Params: 
    * @Return 
    */
    public int maxSubArray(int[] nums) {
        if(nums==null||nums.length==0){
            return 0;
        }
        int max = nums[0];
        int min = nums[0];
        int sum = nums[0];
        for(int i=1; i<nums.length; i++){
            sum += nums[i];
            if(min<0){
                max = Math.max(max, sum-min);
            }
            else{
                max = Math.max(sum,max);
            }
            min = Math.min(sum, min);
        }
        return max;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 76. 最小覆盖子串
    * @DateTime: 11/10/2024 11:54 PM
    * @Params: 
    * @Return 
    */
    public String minWindow(String s, String t) {
        int[] res = new int[2];
        int max = Integer.MAX_VALUE;
        char[] ts = t.toCharArray();
        int[] cnt = new int[256];
        char[] ss = s.toCharArray();
        for(char c : ts){
            cnt[c]--;
        }
        int n=ts.length;
        for(int l=0,r=0; r<ss.length; r++){
            if(cnt[ss[r]]<0){
                n--;
            }
            cnt[ss[r]]++;
            if(n<=0){
                while(cnt[ss[l]]>0){
                    cnt[ss[l++]]--;
                }
                if(r-l+1<max){
                    res[0] = l;
                    res[1] = r;
                    max = r-l+1;
                }
            }
        }

        return max==Integer.MAX_VALUE?"":s.substring(res[0], res[1]+1);
    }

    public boolean checkAllLargerThanZero(int[] cnt){
        for(int i : cnt){
            if(i<0){
                return false;
            }
        }
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 239. 滑动窗口最大值
    * @DateTime: 11/10/2024 8:13 PM
    * @Params: 
    * @Return 
    */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums==null||nums.length<k){
            return new int[]{};
        }
        int[] result = new int[nums.length-k+1];
        int max = Integer.MIN_VALUE;
        int maxId = -1;
        for(int r=k-1,l=0; r<nums.length; r++){
            if(l<=maxId){
                if(nums[r]>=max){
                    max = nums[r];
                    maxId = r;
                }
            }
            else if(nums[r]>=max-1){
                max = nums[r];
                maxId = r;
            }
            else if(nums[l]>=max-1){
                max = nums[l];
                maxId = l;
            }
            else{
                max = Integer.MIN_VALUE;
                for(int i=l; i<=r; i++){
                    if(nums[i]>=max){
                        max = nums[i];
                        maxId = i;
                    }
                }
            }
            result[l++] = max;
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 560. 和为 K 的子数组
    * @DateTime: 11/10/2024 7:38 PM
    * @Params:
    * @Return
    */
    public int subarraySum(int[] nums, int k) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0,1);
        int sum = 0;
        for(int i=0; i<nums.length; i++){
            int n = nums[i];
            sum += n;
            if(map.containsKey(sum-k)){
                res += map.get(sum-k);
            }
            if(map.containsKey(sum)){
                map.put(sum, map.get(sum)+1);
            }
            else{
                map.put(sum,1);
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 438. 找到字符串中所有字母异位词
    * @DateTime: 11/10/2024 7:26 PM
    * @Params: 
    * @Return 
    */
    public List<Integer> findAnagrams(String s, String p) {
        if(s==null||p==null||p.length()==0||p.length()>s.length()){
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        int[] cnt = new int[26];
        char[] p_char = p.toCharArray();
        for(char c : p_char){
            cnt[c-'a']++;
        }
        char[] s_char = s.toCharArray();
        int l = 0, r = p_char.length-1;
        for(int i=0; i<=r; i++){
            char cur = s_char[i];
            cnt[cur-'a']--;
        }
        if(checkAllZero(cnt)){
            res.add(l);
        }
        for(r++;r<s_char.length; l++,r++){
            char left = s_char[l];
            char right = s_char[r];
            cnt[left-'a']++;
            cnt[right-'a']--;
            if(checkAllZero(cnt)){
                res.add(l+1);
            }
        }
        return res;
    }

    public boolean checkAllZero(int[] cnt){
        if(cnt==null||cnt.length==0){
            return true;//
        }
        for(int i=0; i<cnt.length; i++){
            if(cnt[i]!=0){
                return false;
            }
        }
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 3. 无重复字符的最长子串
    * @DateTime: 11/10/2024 7:11 PM
    * @Params: 
    * @Return 
    */
    public int lengthOfLongestSubstring(String s) {
        if(s==null||s.length()==0){
            return 0;
        }
        int[] exist = new int[256];
        Arrays.fill(exist, -1);
        char[] ss = s.toCharArray();
        int pre = 0;
        int res = 0;
        for(int i=0; i<ss.length; i++){
            char cur = ss[i];
            int preInd = exist[cur];
            if(preInd>=pre){
                res = Math.max(res, i-pre);
                pre = preInd+1;
            }
            exist[cur] = i;
        }
        res = Math.max(res, ss.length-pre);
        return res;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 42. 接雨水
    * @DateTime: 11/10/2024 12:28 AM
    * @Params: 
    * @Return 
    */
    public int trap(int[] height) {
        int res = 0;
        int l=0, r=height.length-1;
        int maxHeight = 0;
        while(l<r){
            int left = height[l];
            int right = height[r];
            int tempL = l, tempR=r;
            if(left<right){
                maxHeight = Math.max(maxHeight, left);
                l++;
            }
            else if(left>right){
                maxHeight = Math.max(maxHeight, right);
                r--;
            }
            else{
                maxHeight = Math.max(maxHeight, left);
                l++;r--;
            }
            if(l!=tempL&&l!=tempR){
                res += Math.max(0, (maxHeight-height[l]));
            }
            if(r!=tempR&&r!=tempL&&r!=l){
                res += Math.max(0, (maxHeight-height[r]));
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 15. 三数之和
    * @DateTime: 11/9/2024 11:46 PM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        Set<Integer> set = new HashSet<>();
        for(int i=0; i<nums.length; i++){
            int a = nums[i];
            if(set.contains(a)){
                continue;
            }
            set.add(a);
            int l=i+1, r=nums.length-1;
            while(l<r){
                int sum = nums[l]+nums[r];
                if(sum==-a){
                    List<Integer> list = new ArrayList<>();
                    list.add(a);
                    list.add(nums[l]);
                    list.add(nums[r]);
                    result.add(list);
                    l++;r--;
                    while(l<r&&nums[l]==nums[l-1]){
                        l++;
                    }
                    while(r>l&&nums[r]==nums[r+1]){
                        r--;
                    }
                }
                else if(sum>-a){
                    r--;
                }
                else{
                    l++;
                }
            }
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 11. 盛最多水的容器
    * @DateTime: 11/9/2024 11:32 PM
    * @Params: 
    * @Return 
    */
    public int maxArea2(int[] height) {
        int left = 0, right = height.length-1;
        int result = 0;
        while(left<right){
            int h = Math.min(height[left], height[right]);
            result = Math.max(result, h*(right-left));
            if(height[left]>height[right]){
                int temp_r = right;
                while(left<right&&height[right]<=height[temp_r]){
                    right--;
                }
            }
            else{
                int temp_l = left;
                while(left<right&&height[temp_l]>=height[left]){
                    left++;
                }
            }
        }
        return result;
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 283. 移动零
    * @DateTime: 11/9/2024 12:31 AM
    * @Params: 
    * @Return 
    */
    public void moveZeroes(int[] nums) {
        int cnt_0 = 0;
        for(int i=0; i<nums.length; i++){
            if(nums[i]==0){
                cnt_0++;
            }
            else{
                nums[i-cnt_0] = nums[i];
            }
        }
        for(int i=nums.length-1; cnt_0>0; cnt_0--,i--){
            nums[i]=0;
        }
    }

    public void swap(int[] nums, int l, int r){
        int temp = nums[l];
        nums[l] = nums[r];
        nums[r] = temp;
    }


    /**
    * @Author: Wang Xinxiang
    * @Description: 128. 最长连续序列
    * @DateTime: 11/9/2024 12:17 AM
    * @Params:
    * @Return
    */
    public int longestConsecutive(int[] nums) {
        if(nums==null){
            return 0;
        }
        Arrays.sort(nums);
        int cur = 1;
        int result = 1;
        for(int i=1; i<nums.length; i++){
            if(nums[i]==nums[i-1]){
                continue;
            }
            else if(nums[i]==nums[i-1]+1){
                cur++;
            }
            else{
                result = Math.max(result, cur);
                cur = 1;
            }
        }
        result = Math.max(result, cur);
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description:49. 字母异位词分组
    * @DateTime: 11/8/2024 11:57 PM
    * @Params:
    * @Return
    */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> count = new HashMap<>();
        for(int i=0; i<strs.length; i++){
            char[] ss = strs[i].toCharArray();
            Arrays.sort(ss);
            String cnt = new String(ss);
            List<String> ls = count.getOrDefault(cnt, new ArrayList<>());
            ls.add(strs[i]);
            count.put(cnt, ls);
        }
        return new ArrayList<List<String>>(count.values());
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 
    * @DateTime: 11/8/2024 11:58 PM
    * @Params: 
    * @Return 
    */
    public boolean plantFlowers(int[] flowerbed, int n) {
        // write code here
        if(n==0){
            return true;
        }
        int sum=0;
        for(int i=0; i<flowerbed.length; i++){
            if(plantable(flowerbed,i)){
                sum++;
                flowerbed[i] = 1;
            }
        }
        return sum>=n;
    }

    public boolean plantable(int[] flowerbed, int m){
        if(flowerbed[m]==1){
            return false;
        }
        if(m>0&&flowerbed[m-1]==1){
            return false;
        }
        if(m<flowerbed.length-1&&flowerbed[m+1]==1){
            return false;
        }
        return true;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 
    * @DateTime: 9/3/2024 4:29 AM
    * @Params: 
    * @Return 
    */

    public String mostFrequentSubstring(String s, int k) {
        // write code here
        if(s==null||s.length()<k||k==0){
            return "";
        }
        Map<String, Integer> map = new HashMap<>();
        int max = Integer.MIN_VALUE;
        String res = "";
        for(int i=0; i<=s.length()-k; i++){
            String cur = s.substring(i, i+k);
            int n = map.getOrDefault(cur,0)+1;
            if(max<n){
                max=n;
                res = cur;
            }
            else if(max==n){
                res = getSmallerString(cur, res);
            }
            map.put(cur, n);
        }
        return res;
    }

    public String getSmallerString(String a, String b){
        for(int i=0; i<a.length(); i++){
            if(a.charAt(i)<b.charAt(i)){
                return a;
            }
            else if(a.charAt(i)>b.charAt(i)){
                return b;
            }
        }
        return a;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description:
    * @DateTime: 9/3/2024 4:13 AM
    * @Params:
    * @Return
    */
    public class TicketCache{
        Map<Integer, int[]> map;
        PriorityQueue<int[]> queue;
        int Capacity;
        public TicketCache(int Capacity){
            this.Capacity = Capacity;
            this.map = new HashMap<>();
            this.queue = new PriorityQueue<>((a,b)->a[1]-b[1]);
        }

        public int getTicket(int ticketId){
            if(map.containsKey(ticketId)){
                int[] t = this.map.get(ticketId);
                t[1]++;
                this.map.put(ticketId, t);
                this.queue.add(new int[]{ticketId, t[1]});
                return t[0];
            }
            return -1;
        }

        public void putTicket(int ticket, int info){
            if(this.Capacity==0){
                return;
            }
            if(map.size()==this.Capacity){
                while(!this.queue.isEmpty()){
                    int[] cur = this.queue.poll();
                    if(this.map.get(cur[0])[1]==cur[1]){
                        this.map.remove(cur[0]);
                        break;
                    }
                }
            }
            this.map.put(ticket, new int[]{info,1});
            queue.add(new int[]{ticket, 1});
        }
    }

    public int[] performOperation (String[] operations, int[][] data, int capacity) {
        // write code here
        TicketCache ticketCache = new TicketCache(capacity);
        List<Integer> res = new ArrayList<>();
        for(int i=0; i<operations.length; i++){
            if(operations[i].equals("putTicket")){
                ticketCache.putTicket(data[i][0], data[i][1]);
            }
            else if(operations[i].equals("getTicket")){
                res.add(ticketCache.getTicket(data[i][0]));
            }
        }
        int[] ress = new int[res.size()];
        for(int i=0; i<ress.length; i++){
            ress[i] = res.get(i);
        }
        return ress;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 32. 最长有效括号
    * @DateTime: 8/28/2024 2:04 PM
    * @Params: 
    * @Return 
    */
    public int longestValidParentheses(String s) {
        if(s==null||s.length()==0){
            return 0;
        }
        int res = 0;
        char[] prefix = s.toCharArray();
        int[] dp = new int[prefix.length];
        for(int i=1; i<prefix.length; i++){
            if(prefix[i]==')'){
                if(prefix[i-1]=='('){
                    dp[i] = 2;
                    if(i>2){
                        dp[i] += dp[i-2];
                    }
                    res = Math.max(res, dp[i]);
                }
                else{
                    int l = dp[i-1];
                    if(i-l-1>=0&&prefix[i-l-1]=='('){
                        dp[i] = dp[i-1]+2;
                        if(i-l-2>=0){
                            dp[i] += dp[i-l-2];
                        }
                        res = Math.max(res, dp[i]);
                    }
                }
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 349. 两个数组的交集
    * @DateTime: 8/22/2024 9:31 PM
    * @Params: 
    * @Return 
    */
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        for(int num : nums1){
            set1.add(num);
        }
        Set<Integer> set2 = new HashSet<>();
        for(int num:nums2){
            if(set1.contains(num)){
                set2.add(num);
            }
        }
        int[] res = new int[set2.size()];
        int i=0;
        for(int num:set2){
            res[i++] = num;
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 436. 寻找右区间
    * @DateTime: 8/22/2024 8:42 PM
    * @Params: 
    * @Return 
    */
    public int[] findRightInterval(int[][] intervals) {
        //boolean[] deleted = new boolean[intervals.length];
        int[][] copy = new int[intervals.length][2];
        for(int i=0; i<copy.length; i++){
            copy[i][0] = intervals[i][0];
            copy[i][1] = i;
        }
        Arrays.sort(copy, (a,b)->a[0]-b[0]);
        int[] res = new int[intervals.length];
        Arrays.fill(res,-1);
        for(int i=0; i<intervals.length; i++){
            int[] cur = intervals[i];
            int l=0, r=copy.length-1;
            while(l<r){
                int mid = (l+r)/2;
                int[] temp = copy[mid];
                if(temp[0]==cur[1]){
                    res[i] = temp[1];
                    break;
                }
                else if(temp[0]<cur[1]){
                    l = mid+1;
                }
                else{
                    r=mid;
                }
            }
            if(res[i]==-1&&(r<copy.length-1||copy[r][0]>=cur[1])){
                res[i] = copy[r][1];
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 162. 寻找峰值
    * @DateTime: 8/22/2024 8:32 PM
    * @Params: 
    * @Return 
    */
    public int findPeakElement(int[] nums) {
        if(nums==null||nums.length==0){
            return -1;
        }
        if(nums.length==1){
            return 0;
        }
        if(nums[0]>nums[1]){
            return 0;
        }
        if(nums[nums.length-1]>nums[nums.length-2]){
            return nums.length-1;
        }
        int l=1, r=nums.length-2;
        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid]>nums[mid-1]&&nums[mid]>nums[mid+1]){
                return mid;
            }
            if(nums[mid]>nums[mid-1]){
                l = mid+1;
            }
            if(nums[mid]>nums[mid+1]){
                r=mid-1;
            }
        }
        return -1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 410. 分割数组的最大值
    * @DateTime: 8/22/2024 3:04 AM
    * @Params: 
    * @Return 
    */
    public int splitArray(int[] nums, int k) {
        double sum = 0;
        for(int num:nums){
            sum += num;
        }
        double devide = sum/k;
        int temp=0;
        int max = 0;
        for(int i=0; i<nums.length; i++){
            if(temp==0){
                temp += nums[i];
            }
            else{
                int cur = temp+nums[i];
                if(cur>devide){
                    if(devide-temp<cur-devide){
                        max = Integer.max(temp, max);
                        temp = nums[i];
                    }
                    else{
                        max = Integer.max(cur, max);
                        temp = 0;
                    }
                }
                else{
                    temp = cur;
                }
            }
        }
        return max;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 122. 买卖股票的最佳时机 II
    * @DateTime: 8/22/2024 3:00 AM
    * @Params: 
    * @Return 
    */
    public int maxProfit(int[] prices) {
        if(prices==null||prices.length<2){
            return 0;
        }
        int r=1;
        int profit = 0;
        for(;r<prices.length; r++){
            if(prices[r]>prices[r-1]){
                profit += prices[r]-prices[r-1];
            }
        }
        return profit;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 455. 分发饼干
    * @DateTime: 8/22/2024 2:52 AM
    * @Params: 
    * @Return 
    */
    public int findContentChildren(int[] g, int[] s) {
        if(s==null||s.length==0||g==null||g.length==0){
            return 0;
        }
        Arrays.sort(g);
        Arrays.sort(s);
        int res=0;
        for(int i=0,j=0; i<g.length&&j<s.length; j++){
            if(g[i]<=s[j]){
                res++;
                i++;
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 409. 最长回文串
    * @DateTime: 8/22/2024 2:44 AM
    * @Params: 
    * @Return 
    */
    public int longestPalindrome1(String s) {
        if(s==null||s.length()==0){
            return 0;
        }
        char[] ss = s.toCharArray();
        Map<Character, Integer> map = new HashMap<>();
        for(char c : ss){
            map.put(c, map.getOrDefault(c,0)+1);
        }
        int l = 0;
        for(char c : map.keySet()){
            int cnt = map.get(c);
            if(cnt>=2){
                l += cnt-cnt%2;
            }
        }
        if(l<ss.length){
            l++;
        }
        return l;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 322. 零钱兑换
    * @DateTime: 8/22/2024 2:32 AM
    * @Params: 
    * @Return 
    */
    public int coinChange(int[] coins, int amount) {
        if(amount==0){
            return 0;
        }
        int[] b = new int[amount+1];
        Arrays.fill(b, Integer.MAX_VALUE);
        for(int i=0; i<coins.length; i++){
            if(coins[i]<=amount){
                b[coins[i]]=1;
            }
        }
        for(int i=1; i<amount; i++){
            if(b[i]==Integer.MAX_VALUE){
                continue;
            }
            for(int coin:coins){
                if(i+coin<=amount&&i+coin>0){
                    b[i+coin] = Math.min(b[i]+1, b[i+coin]);
                }
            }
        }
        return b[amount]==Integer.MAX_VALUE?-1:b[amount];
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 127. 单词接龙
    * @DateTime: 8/22/2024 2:21 AM
    * @Params: 
    * @Return 
    */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Queue<String> queue = new ArrayDeque<>();
        queue.add(beginWord);
        int cnt=0;
        while(!queue.isEmpty()){
            cnt++;
            int size = queue.size();
            for(int i=0; i<size; i++){
                String cur = queue.poll();
                if(cur.equals(endWord)){
                    return cnt;
                }
                for(int j=wordList.size()-1; j>=0; j--){
                    if(shortTransfer(cur,wordList.get(j))){
                        queue.add(wordList.get(j));
                        wordList.remove(j);
                    }
                }
            }
        }
        return 0;
    }

    public boolean shortTransfer(String a, String b){
        if(a.length()!=b.length()){
            return false;
        }
        int cnt=0;
        for(int i=0; i<a.length(); i++){
            if(a.charAt(i)!=b.charAt(i)){
                cnt++;
                if(cnt>1){
                    return false;
                }
            }
        }
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 107. 二叉树的层序遍历 II
    * @DateTime: 8/22/2024 2:01 AM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if(root==null){
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        List<Integer> sizes = new ArrayList<>();
        Stack<Integer> stack1 = new Stack<>();
        while(!queue.isEmpty()){
            int size = queue.size();
            sizes.add(size);
            for(int i=0; i<size; i++){
                TreeNode cur = queue.poll();
                if(cur.right!=null){
                    queue.add(cur.right);
                }
                if(cur.left!=null){
                    queue.add(cur.left);
                }
                stack1.push(cur.val);
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for(int i=sizes.size()-1; i>=0; i--){
            List<Integer> list = new ArrayList<>();
            int l = sizes.get(i);
            System.out.println(l);
            for(int j=0; j<l&&!stack1.isEmpty(); j++){
                list.add(stack1.pop());
            }
            res.add(list);
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 529. 扫雷游戏
    * @DateTime: 8/21/2024 10:08 PM
    * @Params: 
    * @Return 
    */
    char[] set = new char[]{'0','1','2','3','4','5','6','7','8'};
    public char[][] updateBoard(char[][] board, int[] click) {

        int m=click[0], n=click[1];
        if(board[m][n]=='M'){
            board[m][n]='X';
            return board;
        }
        int cnt = countM(board, m, n);
        if(cnt!=0){
            board[m][n] = set[cnt]; //
            return board;
        }
        boolean[][] visited = new boolean[board.length][board[0].length];
        flooddfs(board, m, n, visited);
        return board;
    }
    void flooddfs(char[][] board, int m, int n, boolean[][] visited){
        if(m>=board.length||m<0||n<0||n>=board[0].length){
            return;
        }
        if(visited[m][n]){
            return;
        }
        visited[m][n] = true;
        int cnt = countM(board, m, n);
        if(cnt==0){
            board[m][n] = 'B';
            for(int i=m-1; i<=m+1; i++){
                for(int j=n-1; j<=n+1; j++){
                    flooddfs(board, i, j, visited);
                }
            }
        }
        else{
            board[m][n] = set[cnt];
        }
    }
    int countM(char[][] board, int m, int n){
        int cnt=0;
        for(int i=m-1; i<=m+1; i++){
            for(int j=n-1; j<=n+1; j++){
                if(i>=board.length||i<0||j<0||j>=board[0].length){
                    continue;
                }
                if(board[i][j]=='M'){
                    cnt++;
                }
            }
        }
        return cnt;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 130. 被围绕的区域
    * @DateTime: 8/21/2024 9:17 PM
    * @Params: 
    * @Return 
    */
    int[][] d = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    public void solve(char[][] board) {
        if(board==null||board.length==0||board[0].length==0){
            return;
        }
        boolean[][] visted = new boolean[board.length][board[0].length];
        for(int i=1; i<board.length-1; i++){
            for(int j=1; j<board[0].length-1; j++){
                if(visted[i][j]||board[i][j]=='X'){
                    continue;
                }
                boolean res = dfs(board, i, j, visted);
                if(res){
                    flood(board, i, j);
                }
            }
        }
    }
    public boolean dfs(char[][] board, int m, int n, boolean[][] visited){
        if(m<0||n<0||m>=board.length||n>=board[0].length){
            return false;
        }
        if(visited[m][n]){
            return true;
        }
        visited[m][n]=true;
        if(board[m][n]=='X'){
            return true;
        }
        boolean res = true;
        for(int i=0; i<4; i++){
            res = dfs(board, m+d[i][0], n+d[i][1], visited)&&res;
        }
        return res;
    }
    public void flood(char[][] board, int m, int n){
        if(m<0||n<0||m>=board.length||n>=board[0].length){
            return;
        }
        if(board[m][n]=='X'){
            return;
        }
        board[m][n] = 'X';
        for(int i=0; i<4; i++){
            flood(board, m+d[i][0], n+d[i][1]);
        }
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 365. 水壶问题
    * @DateTime: 8/21/2024 9:17 PM
    * @Params: 
    * @Return 
    */
    public boolean canMeasureWater(int x, int y, int target) {
        Queue<int[]> queue = new ArrayDeque<>();
        queue.add(new int[]{0,0});
        boolean[][] visited = new boolean[x+1][y+1];
        while(!queue.isEmpty()){
            int[] cur = queue.poll();
            int rX = cur[0];
            int rY = cur[1];
            if(rX==target||rY==target||rX+rY==target){
                return true;
            }
            if(visited[rX][rY]){
                continue;
            }
            visited[rX][rY]=true;
            queue.add(new int[]{0,rY});
            queue.add(new int[]{rX,0});
            queue.add(new int[]{x,rY});
            queue.add(new int[]{rX,y});
            queue.add(new int[]{rX-Math.min(rX, y-rY),rY+Math.min(rX, y-rY)});
            queue.add(new int[]{rX+Math.min(rY, x-rX),rY-Math.min(rY, x-rX)});
        }
        return false;
    }


    /**
    * @Author: Wang Xinxiang
    * @Description: 104. 二叉树的最大深度
    * @DateTime: 8/21/2024 8:17 PM
    * @Params: 
    * @Return 
    */
    public int maxDepth(TreeNode root) {
        if(root==null){
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right))+1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 493. 翻转对
    * @DateTime: 8/21/2024 1:05 AM
    * @Params: 
    * @Return 
    */
    public int reversePairs(int[] nums) {
        int cnt = 0;
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0; i<nums.length; i++){
            map.put(nums[i], map.getOrDefault(nums[i],0)+1);
        }
        int min = nums[0];
        for(int i=0; i<nums.length; i++){
            map.put(nums[i],map.get(nums[i])-1);
            if(nums[i]<min){
                continue;
            }
            for(int n : map.keySet()){
                long temp = n;
                temp *= 2;
                if(temp<nums[i]){
                    cnt += map.get(n);
                    min = Math.min(min, nums[i]);
                }
            }
        }
        return cnt;
    }
    public int binarySearch(long[] nums, int target, boolean[] deleted){
        int l=0,r=nums.length-1;
        while(l<=r){
            int mid = (l+r)/2;
            if(nums[mid]==target){
                while(mid>=0&&nums[mid]==target&&deleted[mid]){
                    mid--;
                }
                if(mid<0||nums[mid]!=target){
                    mid++;
                }
                while(mid<nums.length&&nums[mid]==target&&!deleted[mid]){
                    mid++;
                }
                if(mid>=nums.length||nums[mid]!=target||deleted[mid]){
                    mid--;
                }
                return mid;
            }
            if(nums[mid]<target){
                l=mid+1;
            }
            else{
                r=mid-1;
            }
        }
        return -1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 324. 摆动排序 II
    * @DateTime: 8/21/2024 12:13 AM
    * @Params: 
    * @Return 
    */
    public void wiggleSort(int[] nums) {
        int[] clone = nums.clone();
        Arrays.sort(clone);
        int mid = (nums.length+1)/2 - 1;
        int r = nums.length-1;
        for(int i=0; i<nums.length; i++){
            nums[i] = clone[mid--];
            if(i+1<nums.length){
                nums[i+1] = clone[r--];
            }
        }

    }
    public void reverse1(int[] nums, int l, int r){
        while(l<r){
            swap(nums, l++, r--);
        }
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 60. 排列序列
    * @DateTime: 8/20/2024 11:26 PM
    * @Params: 
    * @Return 
    */
    public String getPermutation(int n, int k) {
        StringBuilder sb = new StringBuilder();
        List<Integer> list = new ArrayList<>();
        int times = 1;
        for(int i=1; i<=n; i++){
            list.add(i);
            times *= i;
        }
        int i=n;
        while(!list.isEmpty()){
            times = times/i;
            i--;
            int ind = k/times;
            k = k%times;
            if(k==0){
                ind--;
                k=times;
            }
            int cur = list.get(ind);
            sb.append(cur);
            System.out.println(cur);
            list.remove(ind);
        }
        return sb.toString();
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 21. 合并两个有序链表
    * @DateTime: 8/20/2024 11:20 PM
    * @Params: 
    * @Return 
    */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode head = new ListNode();
        ListNode cur = head;
        while(list1!=null&&list2!=null){
            if(list1.val<list2.val){
                cur.next = list1;
                list1 = list1.next;
            }
            else{
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if(list1!=null){
            cur.next = list1;
        }
        else if(list2!=null){
            cur.next = list2;
        }
        return head.next;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 486. 预测赢家
    * @DateTime: 8/20/2024 10:43 PM
    * @Params: 
    * @Return 
    */

    public boolean predictTheWinner(int[] nums) {
        int[][] mem = new int[nums.length][nums.length];
        return getMaxCredits(nums, 0, nums.length-1,mem)>=0;
    }
    public int getMaxCredits(int[] nums, int i, int j, int[][] mem){
        if(i==j){
            return nums[i];
        }
        if(mem[i][j]!=0){
            return mem[i][j];
        }
        int left = nums[j]-getMaxCredits(nums, i, j-1,mem);
        int right = nums[i]-getMaxCredits(nums, i+1, j,mem);
        int res = Math.max(left, right);
        mem[i][j] = res;
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 394. 字符串解码
    * @DateTime: 8/20/2024 6:06 PM
    * @Params: 
    * @Return 
    */
    public String decodeString(String s) {
        if(s==null){
            return null;
        }
        return getSubString(s, 0, s.length()-1);
    }
    public String getSubString(String s, int l, int r){
        if(l>r){
            return "";
        }
        int i=l;
        for(; i<=r&&!(s.charAt(i)<='9'&&s.charAt(i)>='0'); i++);
        String pre = s.substring(l,i);
        int cnt = 0;
        for(;i<=r&&(s.charAt(i)<='9'&&s.charAt(i)>='0'); i++){
            cnt = cnt*10 + s.charAt(i)-'0';
        }
        String mid = "";
        if(i<=r&&s.charAt(i)=='['){
            int cntBrackets = 1;
            i++;
            int start = i;
            for(; i<=r&&cntBrackets>0; i++){
                if(s.charAt(i)=='['){
                    cntBrackets++;
                }
                else if(s.charAt(i)==']'){
                    cntBrackets--;
                }
            }
            mid = getSubString(s, start, i-2);
        }
        String after = getSubString(s,i,r);
        StringBuilder sb = new StringBuilder();
        sb.append(pre);
        for(int m=0; m<cnt; m++){
            sb.append(mid);
        }
        sb.append(after);
        return sb.toString();
    }
    

    /**
    * @Author: Wang Xinxiang
    * @Description: 50. Pow(x, n)
    * @DateTime: 8/17/2024 3:23 AM
    * @Params: 
    * @Return 
    */
    public double myPow(double x, int n) {
        return myPow1(x,n);
    }
    public double myPow1(double x, int n) {
        if(n==0) {
            return 1;
        }
        int cur=0;
        double res=0;
        if(n>0){
            res = x;
            cur = 1;
            while(cur<Integer.MAX_VALUE&&cur<=n/2){
                cur *= 2;
                res *= res;
            }
        }
        else if(n<0){
            res = 1/x;
            cur = -1;
            while(cur>Integer.MIN_VALUE&&cur>=n/2){
                cur *= 2;
                res *= res;
            }
        }
        return res*myPow1(x,n-cur);
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 887. 鸡蛋掉落
    * @DateTime: 8/16/2024 11:02 AM
    * @Params: 
    * @Return 
    */
    public int superEggDrop(int k, int n){
        int[][] dp = new int[n+1][k+1];
        int m=0;
        while(dp[m][k]<n) {
            m++;
            for (int i = 0; i < k; i++) {
                dp[m][k] = dp[m-1][k-1]+dp[m-1][k];
            }
        }
        return m;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 787. K 站中转内最便宜的航班
    * @DateTime: 8/15/2024 10:37 PM
    * @Params: 
    * @Return 
    */
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        //!!defence
        int[] cost = new int[n];
        Arrays.fill(cost, Integer.MAX_VALUE);
        cost[src] = 0;

        for(int i=0; i<=k; i++){
            int[] temp = Arrays.copyOf(cost, n);
            for(int[] flight:flights){
                if(cost[flight[0]]!=Integer.MAX_VALUE){
                    temp[flight[1]] = Math.min(cost[flight[1]], cost[flight[0]]+flight[2]);
                }
            }
            cost = Arrays.copyOf(temp, n);
        }
        if(cost[dst]==Integer.MAX_VALUE){
            return -1;
        }
        return cost[dst];
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 785. 判断二分图
    * @DateTime: 8/15/2024 10:27 PM
    * @Params: 
    * @Return 
    */
    public boolean isBipartite(int[][] graph) {
        boolean[] sides = new boolean[graph.length];//true:side A; false:side B
        boolean[] visted = new boolean[graph.length];//node i has been visited or not
        for(int i=0; i<graph.length; i++){
            if(!visted[i]&&!assignNodeSides(graph, sides, visted, true, i)){
                return false;
            }
        }
        return true;
    }

    public boolean assignNodeSides(int[][] graph, boolean[] sides, boolean[] visted, boolean side, int cur){
        if(visted[cur]){
            return sides[cur]==side;
        }
        visted[cur] = true;
        sides[cur] = side;
        boolean res = true;
        for(int n : graph[cur]){
            res = res&&assignNodeSides(graph, sides, visted, !side, n);
            if(!res){
                return false;
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 797. 所有可能的路径
    * @DateTime: 8/15/2024 10:16 PM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        if(graph==null||graph.length==0){
            return null;//!!
        }
        List<List<Integer>> res = new ArrayList<>();
        allPathsSourceTarget(graph, 0, new ArrayList<Integer>(), res);
        return res;
    }

    public void allPathsSourceTarget(int[][] graph, int m, List<Integer> temp, List<List<Integer>> res) {
        temp.add(m);
        if(m==graph.length-1){
            List<Integer> cur = new ArrayList<>();
            for(int node : temp){
                cur.add(node);
            }
            res.add(cur);
        }
        for(int n : graph[m]){
            allPathsSourceTarget(graph, n, temp, res);
        }
        temp.remove(temp.size()-1);
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 407. 接雨水 II
    * @DateTime: 8/14/2024 3:53 PM
    * @Params: 
    * @Return 
    */
    int[][] direction = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    public int trapRainWater(int[][] heightMap) {
        int l1 = heightMap.length, l2=heightMap[0].length;
        boolean[][] visited = new boolean[l1][l2];
        int[][] heights = new int[l1][l2];
        int res = 0;
        for(int i=1; i<l1-1; i++){
            for(int j=1; j<l2-1; j++){
                if(heights[i][j]==0){
                    trapRainWater(heightMap, i, j, new boolean[l1][l2], heights);
                }
                res += Math.max(0,heights[i][j]-heightMap[i][j]);
            }
        }

        return res;
    }
    public void trapRainWater(int[][] heightMap,int m, int n, boolean[][] visted, int[][] heights) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b)->a[0]-b[0]);
        queue.add(new int[]{heightMap[m][n], m, n});
        Queue<int[]> queue1 = new ArrayDeque<>();
        int[] max = queue.peek();
        visted[m][n] = true;
        int j=0, end=0;
        while(!queue.isEmpty()){
            int[] cur = queue.poll();
            j++;
            if(max[0]<cur[0]){
                max = cur;
                end=j;
            }
            queue1.add(cur);
            if(cur[1]==0||cur[1]==heightMap.length-1||cur[2]==0||cur[2]==heightMap[0].length-1){
                break;
            }
            for(int i=0; i<4; i++){
                int mm=cur[1]+direction[i][0];
                int nn=cur[2]+direction[i][1];
                if(!visted[mm][nn]){
                    queue.add(new int[]{heightMap[mm][nn], mm, nn});
                }
                visted[mm][nn] = true;
            }
        }
        while(!queue1.isEmpty()&&end-->1){
            int[] cur = queue1.poll();
            heights[cur[1]][cur[2]] = max[0];
        }
        while(!queue1.isEmpty()){
            int[] cur = queue1.poll();
            heights[cur[1]][cur[2]] = -1;
        }
        while(!queue.isEmpty()){
            int[] cur = queue.poll();
            heights[cur[1]][cur[2]] = -1;
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 4. 寻找两个正序数组的中位数
    * @DateTime: 8/14/2024 3:36 PM
    * @Params: 
    * @Return 
    */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int total = nums1.length+nums2.length;
        int m1 = total/2;
        int m2 = (total-1)/2;
        int l=0, r=0;
        long res = 0;
        int i=0;
        for(; i<=m2&&l<nums1.length&&r<nums2.length; i++){
            if(nums1[l]<=nums2[r]){
                res = nums1[l];
                l++;
            }
            else{
                res = nums2[r];
                r++;
            }
        }
        for(; i<=m2&&l<nums1.length; i++){
            res = nums1[l];
            l++;
        }
        for(; i<=m2&&r<nums2.length; i++){
            res = nums2[r];
            r++;
        }
        if(m1==m2){
            return res;
        }
        if(l<nums1.length&&r<nums2.length){
            if(nums1[l]<=nums2[r]){
                res += nums1[l];
            }
            else{
                res += nums2[r];
            }
        }
        else if(l<nums1.length){
            res += nums1[l];
        }
        else{
            res += nums2[r];
        }
        return res/2;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 207. 课程表
    * @DateTime: 8/13/2024 5:57 PM
    * @Params: 
    * @Return 
    */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> dependency = new HashMap<>();
        for(int[] n : prerequisites){
            List<Integer> temp = dependency.getOrDefault(n[0], new ArrayList<>());
            temp.add(n[1]);
            dependency.put(n[0], temp);
        }
        boolean[] visited = new boolean[numCourses];
        boolean[] checked = new boolean[numCourses];
        for(int m:dependency.keySet()){
            if(checked[m]){
                continue;
            }
            if(!isCycled(dependency, m, visited, checked)){
                return false;
            }
        }
        return true;
    }

    public boolean isCycled(Map<Integer, List<Integer>> dependency, int m, boolean[] visited, boolean[] checked){
        if(visited[m]){
            return false;
        }
        if(checked[m]){
            return true;
        }
        if(!dependency.containsKey(m)){
            checked[m] = true;
            return true;
        }
        visited[m] = true;
        List<Integer> temp = dependency.get(m);
        for(int n:temp){
            if(!isCycled(dependency, n, visited, checked)){
                return false;
            }
        }
        visited[m] = false;
        checked[m] = true;
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 547. 省份数量
    * @DateTime: 8/13/2024 5:32 PM
    * @Params: 
    * @Return 
    */
    public int findCircleNum(int[][] isConnected) {
        if(isConnected==null){
            return 0;
        }
        int result = 0;
        boolean[] visted = new boolean[isConnected.length];
        for(int i=0; i<isConnected.length; i++){
            result += traverseConnections(isConnected, i, visted);
        }
        return result;
    }
    public int traverseConnections(int[][] isConnected, int m, boolean[] visted){
        if(visted[m]){
            return 0;
        }
        visted[m] = true;
        for(int i=0; i<isConnected[0].length; i++){
            if(isConnected[m][i]==1){
                traverseConnections(isConnected, i, visted);
            }
        }
        return 1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 133. 克隆图
    * @DateTime: 8/13/2024 4:35 PM
    * @Params: 
    * @Return 
    */
    Map<Node, Node> map = new HashMap<>();
    public Node cloneGraph(Node node) {
        if(node==null){
            return null;
        }
        if(map.containsKey(node)) {
            return map.get(node);
        }
        Node root = new Node(node.val);
        map.put(node, root);
        List<Node> nodes = new ArrayList<>();
        for(Node n : node.neighbors) {
            nodes.add(cloneGraph(n));
        }
        root.neighbors = nodes;
        return root;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 无向连接图
    * @DateTime: 8/13/2024 4:34 PM
    * @Params: 
    * @Return 
    */
    class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
    /**
     * @Author: Wang Xinxiang
     * @Description: 315. 计算右侧小于当前元素的个数
     * @DateTime: 8/12/2024 6:49 AM
     * @Params:
     * @Return
     */
    int[] a;
    public List<Integer> countSmaller(int[] nums) {
        discretization(nums);
        List<Integer> list = new ArrayList<>();
        int[] b = new int[a.length];
        for(int i=nums.length-1; i>=0; i--){
            int j = binarySearch(a, nums[i]);
            //!!!!
            list.add(0,b[j]);
            for(j++; j<a.length; j++){
                b[j]++;
            }
        }
        return list;
    }

    public int binarySearch(int[] a, int x){
        int l=0, r=a.length-1;
        while(l<=r){
            int mid = (l+r)/2;
            if(a[mid]==x){
                return mid;
            }
            else if(a[mid]<x){
                l=mid+1;
            }
            else{
                r=mid-1;
            }
        }
        return -1;
    }
    public void discretization(int[] nums){
        Set<Integer> set = new HashSet<>();
        for(int n : nums){
            set.add(n);
        }
        int i=0;
        a = new int[set.size()];
        for(int n : set){
            a[i++] = n;
        }
        Arrays.sort(a);
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 406. 根据身高重建队列
    * @DateTime: 8/11/2024 11:27 AM
    * @Params: 
    * @Return 
    */
    public int[][] reconstructQueue(int[][] people) {
        PriorityQueue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] ints, int[] t1) {
                if(ints[0]!=t1[0]){
                    return t1[0]-ints[0];
                }
                return ints[1] - t1[1];
            }
        });
        for(int[] person:people){
            queue.add(person);
        }
        List<int[]> res = new ArrayList<>();
        for(int i=0; i<people.length; i++){
            int[] person = queue.poll();
            res.add(person[1],person);
        }
        return res.toArray(new int[people.length][2]);
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 386. 字典序排数
    * @DateTime: 8/11/2024 11:11 AM
    * @Params: 
    * @Return 
    */
    public List<Integer> lexicalOrder(int n) {
        if(n<1){
            return new ArrayList<>();
        }
        List<Integer> list = new ArrayList<>();
        for(int j=1; j<=9; j++){
            if(j>n){
                break;
            }
            list.add(j);
            lexicalOrder1(n, j*10, list);
        }
        return list;
    }
    public void lexicalOrder1(int n, int i, List<Integer> list){
        for(int j=0; j<=9; j++){
            if(i+j>n){
                return;
            }
            list.add(i+j);
            lexicalOrder1(n, (i+j)*10, list);
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 200. 岛屿数量
    * @DateTime: 8/11/2024 10:55 AM
    * @Params: 
    * @Return 
    */
    public int numIslands(char[][] grid) {
        int[][] moves = new int[][]{{1,0},{0,1},{0,-1},{-1,0}};
        int num=0;
        for(int i=0; i<grid.length; i++){
            for(int j=0; j<grid[0].length; j++){
                if(grid[i][j]=='1'){
                    num++;
                    visitIsland(grid, i, j, moves);
                }
            }
        }
        return num;
    }
    public void visitIsland(char[][] grid, int m, int n, int[][] moves){
        if(m<0||n<0||m>=grid.length||n>grid[0].length||grid[m][n]=='0'){
            return;
        }
        grid[m][n]='0';
        for(int[] move:moves){
            visitIsland(grid,m+move[0], n+move[1], moves);
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 124. 二叉树中的最大路径和
    * @DateTime: 8/11/2024 10:11 AM
    * @Params: 
    * @Return 
    */
    int max;
    public int maxPathSum(TreeNode root) {
        if(root==null){
            return 0;
        }
        max = Integer.MIN_VALUE;
        maxPathSingleNSum(root);
        return max;
    }

    public int maxPathSingleNSum(TreeNode root){
        if(root==null){
            return 0;
        }
        int sgl = root.val;
        int l = Math.max(0,maxPathSingleNSum(root.left));
        int r = Math.max(0,maxPathSingleNSum(root.right));
        int curSum = sgl+l+r;
        if(max<curSum){
            max = curSum;
        }
        int sub = Math.max(l, r);
        return sgl+sub;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 103. 二叉树的锯齿形层序遍历
    * @DateTime: 8/10/2024 2:08 PM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        if(root!=null){
            queue.add(root);
        }
        List<List<Integer>> result = new ArrayList<>();
        boolean sign = true;
        while(!queue.isEmpty()){
            sign = !sign;
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for(int i=0; i<size; i++){
                TreeNode cur = queue.poll();
                if(sign){
                    list.add(0, cur.val);
                }
                else{
                    list.add(cur.val);
                }
                if(cur.left!=null){
                    queue.add(cur.left);
                }
                if(cur.right!=null){
                    queue.add(cur.right);
                }
            }
            result.add(list);
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 230. 二叉搜索树中第K小的元素
    * @DateTime: 8/10/2024 2:01 PM
    * @Params: 
    * @Return 
    */
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while(cur!=null){
            stack.push(cur);
            cur = cur.left;
        }

        while(!stack.isEmpty()){
            cur = stack.pop();
            k--;
            if(k==0){
                return cur.val;
            }
            if(cur.right!=null){
                cur = cur.right;
                stack.push(cur);
                cur = cur.left;
                while(cur!=null){
                    stack.push(cur);
                    cur = cur.left;
                }
            }
        }
        return -1;//exception
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 236. 二叉树的最近公共祖先
    * @DateTime: 8/10/2024 1:06 PM
    * @Params: 
    * @Return 
    */
    TreeNode result;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //result = null;
        containsOneNode(root, p, q);
        return result;
    }
    public boolean containsOneNode(TreeNode root, TreeNode p, TreeNode q){
        if(root==null){
            return false;
        }
        boolean l = containsOneNode(root.left, p, q);
        boolean r = containsOneNode(root.right, p, q);
        if(root.val==p.val||root.val==q.val){
            if(l||r){
                result = root;
            }
            return true;
        }
        if(l&&r&&result==null){
            result = root;
        }
        return l||r;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 100. 相同的树
    * @DateTime: 8/3/2024 5:43 PM
    * @Params: 
    * @Return 
    */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null&&q==null){
            return true;
        }
        if(p==null||q==null){
            return false;
        }
        if(p.val!=q.val){
            return false;
        }
        return isSameTree(p.left, q.left)&&isSameTree(p.right, q.right);
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 160. 相交链表
    * @DateTime: 8/3/2024 5:34 PM
    * @Params: 
    * @Return 
    */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int m = 0;
        int n = 0;
        ListNode cur = headA;
        for(;cur!=null;cur=cur.next,m++);
        cur = headB;
        for(;cur!=null;cur=cur.next,n++);
        for(;headA!=null&&m>n;headA=headA.next,m--);
        for(;headB!=null&&m<n;headB=headB.next,n--);
        for(;m>0&&headA!=headB;m--,headA=headA.next,headB=headB.next);
        if(m==0){
            return null;
        }
        else{
            return headA;
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 143. 重排链表
    * @DateTime: 8/3/2024 4:38 PM
    * @Params: 
    * @Return 
    */
    public void reorderList(ListNode head) {
        if(head==null||head.next==null){
            return;
        }
        int n=0;
        ListNode cur = head;
        for(;cur!=null;n++, cur = cur.next);
        n = n/2+n%2;
        cur = head;
        for(;n>0;n--,cur = cur.next);
        cur = reverseList(cur);
        mergeList(head,cur);
    }

    public void mergeList(ListNode head, ListNode head2){
        ListNode pre = new ListNode();
        pre.next = head;
        while(head!=null&&head2!=null){
            ListNode next = head.next;
            head.next = head2;
            pre = head2;
            head = next;
            next = head2.next;
            head2.next = head;
            head2 = next;
        }
        if(head==null){
            pre.next = head2;
        }
    }
    public ListNode reverseList(ListNode head){
        ListNode pre = null;
        while(head!=null){
            ListNode next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        while(pre!=null){
            System.out.println(pre.val);
        }
        return pre;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 24. 两两交换链表中的节点
    * @DateTime: 8/3/2024 4:29 PM
    * @Params: 
    * @Return 
    */
    public ListNode swapPairs(ListNode head) {
        if(head==null||head.next==null){
            return head;
        }
        ListNode pre = new ListNode();
        pre.next = head;
        ListNode preHead = pre;
        ListNode cur = head;
        ListNode next = cur.next;
        while(next!=null){
            cur.next = next.next;
            next.next = cur;
            pre.next = next;
            pre = cur;
            cur = cur.next;
            if(cur==null){
                break;
            }
            next = cur.next;
        }
        return preHead.next;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 92. 反转链表 II
    * @DateTime: 8/3/2024 1:09 PM
    * @Params: 
    * @Return 
    */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode cur = head;
        ListNode prehead = new ListNode();
        prehead.next = head;
        for(int i=1; i<left&&cur!=null; i++,cur = cur.next,prehead = prehead.next);
        if(cur==null||cur.next==null){
            return head;
        }
        ListNode curTail = cur;
        ListNode pre = null;
        for(int i = left; i<=right&&cur!=null; i++){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        if(pre!=null){
            curTail.next=cur;
            prehead.next=pre;
        }
        if(left==1){
            return prehead.next;
        }

        return head;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 82. 删除排序链表中的重复元素 II
    * @DateTime: 8/3/2024 12:37 PM
    * @Params: 
    * @Return 
    */
    public ListNode deleteDuplicates(ListNode head) {
        if(head==null||head.next==null){
            return head;
        }
        ListNode pre = new ListNode();
        pre.next = head;
        ListNode preHead = pre;
        ListNode cur = head;
        ListNode next = cur.next;
        boolean deleted = false;
        while(next!=null){
            if(next.val==cur.val){
                pre.next = next.next;
                deleted = true;
            }
            else {
                if(!deleted){
                    pre = pre.next;
                }
                deleted = false;
            }
            cur = next;
            next = next.next;
        }
        return preHead.next;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 61. 旋转链表
    * @DateTime: 8/2/2024 4:22 PM
    * @Params: 
    * @Return 
    */
    public ListNode rotateRight(ListNode head, int k) {
        if(head==null){
            return null;
        }
        int n=0;
        ListNode cur = head;
        ListNode end = new ListNode();
        while(cur!=null){
            end = cur;
            n++;
            cur = cur.next;
        }
        k = k%n;
        if(k==0){
            return head;
        }
        cur = head;
        ListNode pre = new ListNode();
        for(int i=0; i<n-k; i++){
            pre = cur;
            cur = cur.next;
        }
        pre.next=null;
        end.next = head;
        return cur;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 2. 两数相加
    * @DateTime: 8/2/2024 4:01 PM
    * @Params: 
    * @Return 
    */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int jump = 0;
        ListNode result = new ListNode();
        ListNode head = result;
        while(l1!=null&&l2!=null){
            result.next = new ListNode(0);
            result = result.next;
            result.val = l1.val + l2.val + jump;
            jump = result.val/10;
            result.val = result.val%10;
            l1 = l1.next;
            l2 = l2.next;
        }
        while(l1!=null){
            result.next = new ListNode(0);
            result = result.next;
            result.val = l1.val + jump;
            jump = result.val/10;
            result.val = result.val%10;
            l1 = l1.next;
        }
        while(l2!=null){
            result.next = new ListNode(0);
            result = result.next;
            result.val = l2.val + jump;
            jump = result.val/10;
            result.val = result.val%10;
            l2 = l2.next;
        }
        while(jump!=0){
            result.next = new ListNode(0);
            result = result.next;
            result.val = jump;
            jump = result.val/10;
            result.val = result.val%10;
        }
        return head.next;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 718. 最长重复子数组
    * @DateTime: 8/2/2024 2:49 PM
    * @Params: 
    * @Return 
    */
    public int findLength(int[] nums1, int[] nums2) {
        int max = 0;
        int m = nums1.length+1;
        int n = nums2.length+1;
        int[][] dp = new int[m][n];
        for(int i=1; i<m; i++){
            for(int j=1; j<n; j++){
                if(nums1[i-1]==nums2[j-1]){
                    dp[i][j] =dp[i-1][j-1] + 1;
                    max = Math.max(max, dp[i][j]);
                }
            }
        }
        return max;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 632. 最小区间
    * @DateTime: 8/2/2024 1:26 PM
    * @Params: 
    * @Return 
    */
    public int[] smallestRange(List<List<Integer>> nums) {
        PriorityQueue<int[]> mins = new PriorityQueue<>((a,b)->a[0]-b[0]);
        int[] index = new int[nums.size()];
        int max = Integer.MIN_VALUE;
        int n = 0;
        for(int i=0; i<nums.size(); i++){
            n += nums.get(i).size()-1;
            mins.add(new int[]{nums.get(i).get(0), i});
            index[i] = 0;
            max = Math.max(nums.get(i).get(0), max);
        }
        int[] res = new int[2];
        res[0] = mins.peek()[0];
        res[1] = max;
        while(n>0){
            int[] cur = mins.poll();
            List<Integer> list = nums.get(cur[1]);
            if(list.size()==index[cur[1]]+1){
                return res;
            }
            int ind = cur[1];
            index[ind]++;
            max = Math.max(list.get(index[ind]),max);
            mins.add(new int[]{list.get(index[ind]),ind});
            if(res[1]-res[0]>max-mins.peek()[0]){
                res[0] = mins.peek()[0];
                res[1] = max;
            }
        }
        return res;
    }


    /**
    * @Author: Wang Xinxiang
    * @Description: 658. 找到 K 个最接近的元素
    * @DateTime: 8/2/2024 12:28 PM
    * @Params: 
    * @Return 
    */
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        if(arr.length<k){
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        if(x<=arr[0]){
            for(int i=0; i<k; i++){
                res.add(arr[i]);

            }
            return res;
        }
        if(x>=arr[arr.length-1]){
            for(int i=arr.length-k; i<arr.length; i++){
                res.add(arr[i]);

            }
            return res;
        }
        int l = 0, r = arr.length-1;
        while(r-l+1>k){
            if(x-arr[l]<=arr[r]-x){
                r--;
            }
            else{
                l++;
            }
        }
        for(;l<=r;l++){
            res.add(arr[l]);
        }
        return res;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 30. 串联所有单词的子串
    * @DateTime: 8/1/2024 11:30 PM
    * @Params: 
    * @Return 
    */
    public List<Integer> findSubstring(String s, String[] words) {
        Map<String, Integer> differ = new HashMap<>();
        for(int i=0; i<words.length; i++){
            differ.put(words[i],differ.getOrDefault(words[i],0)-1);
        }
        List<Integer> res = new ArrayList<>();
        for(int i=0; i<words[0].length(); i++){
            findSubstring(s, i, differ, words.length, words[0].length(), res);
        }
        return res;
    }

    public void findSubstring(String s, int offset, Map<String, Integer> differ, int n, int size, List<Integer> res){
        if(s.length()-offset < n*size){
            return;
        }
        int cnt = (s.length()-offset)/size;
        String ss = s.substring(offset);
        Map<String, Integer> differ1 = new HashMap<>(differ);
        int index = -1*size;
        for(int i=0; i<n; i++){
            index += size;
            String sub = ss.substring(index, index+size);
            if(differ1.containsKey(sub)){
                differ1.put(sub, differ1.get(sub)+1);
            }
        }
        if(checkComplete(differ1)){
            res.add(offset);
        }
        int preIndex = 0;

        for(int i=n; i<cnt; i++){
            index += size;
            String sub = ss.substring(preIndex, preIndex+size);
            preIndex += size;
            if(differ1.containsKey(sub)){
                differ1.put(sub, differ1.get(sub)-1);
            }
            sub = ss.substring(index, index+size);
            if(differ1.containsKey(sub)){
                differ1.put(sub, differ1.get(sub)+1);
            }
            if(checkComplete(differ1)){
                res.add(preIndex+offset);
            }
        }
    }

    public boolean checkComplete(Map<String, Integer> map){
        for(int s : map.values()){
            if(s!=0){
                return false;
            }
        }
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 480. 滑动窗口中位数
    * @DateTime: 8/1/2024 1:35 PM
    * @Params: 
    * @Return 
    */
    public double[] medianSlidingWindow(int[] nums, int k) {
        if(nums==null||nums.length<k||k==0){
            return new double[]{};//throw?
        }
        PriorityQueue<Integer> small = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer integer, Integer t1) {
                if(integer>t1){
                    return -1;
                }
                return 1;
            }
        });
        PriorityQueue<Integer> large = new PriorityQueue<>();
        Map<Integer, Integer> map = new HashMap<>();
        double[] res = new double[nums.length-k+1];
        for(int i=0; i<k; i++){
            small.add(nums[i]);
        }
        for(int i=0; i<k/2; i++){
            large.add(small.poll());
        }
        res[0] = getMedian(small, large, k);
        int balance = small.size()-large.size();
        for(int i=k; i<nums.length; i++){
            balance = remove(small, large, nums[i-k], balance, map);
            balance = add(small, large, nums[i], balance, map);
            res[i-k+1] = getMedian(small, large,k);
        }
        return res;
    }

    public int add(PriorityQueue<Integer> small, PriorityQueue<Integer> large
            , int n, int balance, Map<Integer, Integer> map){
        if(small.isEmpty()||n<=small.peek()){
            small.add(n);
            balance++;
        }
        else{
            large.add(n);
            balance--;
        }
        if(balance<0){
            small.add(large.poll());
            balance +=2;
        }
        if(balance>1){
            large.add(small.poll());
            balance -=2;
        }
        while(!large.isEmpty()&&map.getOrDefault(large.peek(),0)>0){
            int v = map.get(large.peek());
            map.put(large.poll(), v-1);
        }
        while(!small.isEmpty()&&map.getOrDefault(small.peek(),0)>0){
            int v = map.get(small.peek());
            map.put(small.poll(), v-1);
        }
        return balance;
    }
    public int remove(PriorityQueue<Integer> small, PriorityQueue<Integer> large
            , int n, int balance, Map<Integer, Integer> map){
        map.put(n, map.getOrDefault(n, 0)+1);
        if(n<=small.peek()){
            balance--;
        }
        else{
            balance++;
        }
        if(balance<0){
            small.add(large.poll());
            balance +=2;
        }
        if(balance>1){
            large.add(small.poll());
            balance -= 2;
        }
        while(!small.isEmpty()&&map.getOrDefault(small.peek(),0)>0){
            int v = map.get(small.peek());
            map.put(small.poll(), v-1);
        }
        while(!large.isEmpty()&&map.getOrDefault(large.peek(),0)>0){
            int v = map.get(large.peek());
            map.put(large.poll(), v-1);
        }
        return balance;

    }
    public double getMedian(PriorityQueue<Integer> small, PriorityQueue<Integer> large, int k){
        if(k%2==0){
            return ((double)small.peek())/2+((double)large.peek())/2;
        }
        return small.peek();
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 219. 存在重复元素 II
    * @DateTime: 8/1/2024 1:28 PM
    * @Params: 
    * @Return 
    */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i=0; i<nums.length; i++){
            if(map.containsKey(nums[i])){
                if(i-map.get(nums[i])<=k){
                    return true;
                }
            }
            map.put(nums[i], i);
        }
        return false;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 187. 重复的DNA序列
    * @DateTime: 7/31/2024 7:08 PM
    * @Params: 
    * @Return 
    */
    public List<String> findRepeatedDnaSequences(String s) {
        if(s.length()<10){
            return new ArrayList<>();
        }
        Set<String> seen = new HashSet<>();
        Set<String> added = new HashSet<>();
        List<String> res = new ArrayList<>();
        for(int i=0; i<s.length()-9; i++){
            String temp = s.substring(i,i+10);
            if(seen.contains(temp)){
                if(!added.contains(temp)){
                    added.add((temp));
                    res.add(temp);
                }
            }
            else{
                seen.add(temp);
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 125. 验证回文串
    * @DateTime: 7/31/2024 6:05 PM
    * @Params: 
    * @Return 
    */
    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        char[] valid = s.toString().toCharArray();
        int l=0, r=valid.length-1;
        while(l<r){
            while(!((valid[l]>='a'&&valid[l]<='z')||(valid[l]>='0'&&valid[l]<='9'))&&l<r){
                l++;
            }
            while(!((valid[r]>='a'&&valid[r]<='z')||(valid[r]>='0'&&valid[r]<='9'))&&l<r){
                r--;
            }
            if(valid[l++]!=valid[r--]){
                return false;
            }
        }
        return true;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description:
    * @DateTime: 7/31/2024 5:44 PM
    * @Params:
    * @Return
    */
    public int compareVersion(String version1, String version2) {
        String version11 = version1.trim();
        String[] version1s = version11.split("\\.");
        String version22 = version2.trim();
        String[] version2s = version22.split("\\.");
        if(version1s.length<version2s.length){
            int k = version2s.length-version1s.length;
            for(int i=0; i<k; i++){
                version11 = version11+".0";
            }
            version1s = version11.split("\\.");
        }
        if(version1s.length>version2s.length){
            int k = version1s.length-version2s.length;
            for(int i=0; i<k; i++){
                version22 = version22+".0";
            }
            version2s = version22.split("\\.");
        }
        int m=0, n=0;
        while(m<version1s.length&&n<version2s.length){
            int res = compareVersionPart(version1s[m], version2s[n]);
            if(res!=0){
                return res;
            }
            m++;
            n++;
        }
        return 0;

    }

    public int compareVersionPart(String version1, String version2){
        int v1 = Integer.parseInt(version1);
        int v2 = Integer.parseInt(version2);
        if(v1>v2){
            return 1;
        }
        if(v1==v2){
            return 0;
        }
        return -1;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 189. 轮转数组
    * @DateTime: 7/31/2024 5:28 PM
    * @Params: 
    * @Return 
    */
    public void rotate(int[] nums, int k) {
        k = k%nums.length;
        reverse(nums, 0, nums.length-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, nums.length-1);
        System.out.println();
    }

    public void reverse(int[] nums, int l, int r){
        while(l<r){
            swap(nums, l++, r--);
        }
    }






    /**
    * @Author: Wang Xinxiang
    * @Description: 88. 合并两个有序数组
    * @DateTime: 7/31/2024 5:23 PM
    * @Params: 
    * @Return 
    */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        m--;n--;
        int i=nums1.length-1;
        for(; i>=0&&m>=0&&n>=0; i--){
            if(nums1[m]>nums2[n]){
                nums1[i] = nums1[m--];
            }
            else{
                nums1[i] = nums2[n--];
            }

        }
        while(m>=0){
            nums1[i--] = nums1[m--];
        }
        while(n>=0){
            nums1[i--] = nums2[n--];
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 31. 下一个排列
    * @DateTime: 7/30/2024 11:26 PM
    * @Params: 
    * @Return 
    */
    public void nextPermutation(int[] nums) {
        int l = nums.length-1, r = nums.length-1;
        for(;l>0&&nums[l]<=nums[l-1];l--);
        if(l!=0){
            int i=r;
            for(; i>l-1&&nums[i]<=nums[l-1];i--);
            swap(nums, l-1, i);
        }
        while(l<r){
            swap(nums, l++, r--);
        }
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 26. 删除有序数组中的重复项
    * @DateTime: 7/30/2024 11:21 PM
    * @Params: 
    * @Return 
    */
    public int removeDuplicates(int[] nums) {
        int l=0,r=1;
        for(;r<nums.length;r++){
            if(nums[r]!=nums[l]){
                nums[l+1] = nums[r];
                l++;
            }
        }
        return l+1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 218. 天际线问题
    * @DateTime: 7/30/2024 10:42 PM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for(int[] building:buildings){
            queue.add(building[0]);
            queue.add(building[1]);
        }
        List<List<Integer>> res = new ArrayList<>();
        for(int[] building:buildings){
            if(building[0]==queue.peek()){
                res.add(List.of(building[0],building[2]));
                queue.poll();
            }
            if(building[1]==queue.peek()){
                res.add(List.of(building[1],0));
                queue.poll();
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 264. 丑数 II
    * @DateTime: 7/30/2024 4:27 PM
    * @Params: 
    * @Return 
    */
    public int nthUglyNumber(int n) {
        int product1=1, product2=1, product3=1;
        int p1=0, p2=0, p3=0;
        int[] ugly = new int[n];
        for(int i=0; i<n; i++){
            int min = Math.min(Math.min(product1, product2),product3);
            ugly[i] = min;
            if(min==product1){
                product1 = ugly[p1++]*2;
            }
            if(min==product2){
                product2 = ugly[p2++]*3;
            }
            if(min==product3){
                product3 = ugly[p3++]*5;
            }

        }
        return ugly[n-1];
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 373. 查找和最小的 K 对数字
    * @DateTime: 7/30/2024 3:27 PM
    * @Params: 
    * @Return 
    */
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<int[]> queue = new PriorityQueue<>((a,b)->a[0]-b[0]);
        queue.add(new int[]{nums1[0]+nums2[0],0,0});
        while(k>0){
            int[] cur = queue.poll();
            res.add(List.of(nums1[cur[1]],nums2[cur[2]]));
            k--;
            if(cur[2]==0&&cur[1]+1<nums1.length){
                queue.add(new int[]{nums1[cur[1]+1]+nums2[cur[2]],cur[1]+1,cur[2]});
            }
            if(cur[2]+1<nums2.length){
                queue.add(new int[]{nums1[cur[1]]+nums2[cur[2]+1],cur[1],cur[2]+1});
            }

        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 215. 数组中的第K个最大元素
    * @DateTime: 7/30/2024 11:25 AM
    * @Params: 
    * @Return 
    */
    public int findKthLargest(int[] nums, int k) {
        int max=nums[0], min=nums[0];
        for(int i=0; i<nums.length; i++){
            max = max>nums[i]?max:nums[i];
            min = min<nums[i]?min:nums[i];
        }
        int[] cnt = new int[max-min+1];
        for(int i=0; i<nums.length; i++){
            cnt[nums[i]-min]++;
        }
        for(int i=cnt.length-1; i>=0; i--){
            if(cnt[i]>=k){
                return i+min;
            }
            else{
                k -= cnt[i];
            }
        }
        return min;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 936. 戳印序列
    * @DateTime: 7/30/2024 10:56 AM
    * @Params: 
    * @Return 
    */
    Stack<Integer> stack;
    public int[] movesToStamp(String stamp, String target) {
        stack = new Stack<>();
        ableToStamp(stamp.toCharArray(), target.toCharArray(), 0, target.length()-1);
        return new int[]{};
    }
    public boolean ableToStamp(char[] stamp, char[] target, int start, int end){
        if(end-start<stamp.length-1){
            return false;
        }
        if(start>end){
            return true;
        }
        boolean res = false;
        for(int i=start; i<=end-stamp.length+1; i++){
            int m=i, n=i;
            for(int j=0; j<stamp.length; j++){
                if(stamp[j]==target[i+j]){
                    n++;
                    if(j==stamp.length-1){
                        for(int p=m-1; p<n-1; p++){
                            if(p>0){
                                stack.add(m);
                                res = res||ableToStamp(stamp, target, start, p);
                                stack.pop();
                            }
                        }
                        for(int p=n; p>m; p--){
                            stack.add(m);
                            res = res||ableToStamp(stamp, target, p, end);
                            stack.pop();
                        }
                    }
                }
                else{
                    if(m!=n){
                        for(int p=m-1; p<n-1; p++){
                            if(p>0){
                                stack.add(m);
                                res = res||ableToStamp(stamp, target, start, p);
                                stack.pop();
                            }
                        }
                        for(int p=n; p>m; p--){
                            stack.add(m);
                            res = res||ableToStamp(stamp, target, p, end);
                            stack.pop();
                        }
                        m=n;
                    }
                    else{
                        m++;
                        n++;
                    }
                }
            }
        }
        return res;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 2071. 你可以安排的最多任务数目
    * @DateTime: 7/30/2024 1:41 AM
    * @Params: 
    * @Return 
    */
    public int maxTaskAssign(int[] tasks, int[] workers, int pills, int strength) {
        Queue<Integer> queue = new PriorityQueue<>((a,b)->b-a);
        int res = 0;
        for(int i=0; i<tasks.length; i++){
            queue.add(tasks[i]);
        }
        for(int i=0; i<tasks.length; i++){
            tasks[i] = queue.poll();
        }
        for(int i=0; i<workers.length; i++){
            queue.add(workers[i]);
        }
        for(int i=0; i<workers.length; i++){
            workers[i] = queue.poll();
        }
        for(int i=0,j=0; i<workers.length&&j<tasks.length;){
            if(workers[i]<0){
                i++;
                continue;
            }
            while(j<tasks.length&&tasks[j]==-1){
                j++;
            }
            if(j==tasks.length){
                break;
            }
            if(workers[i]>=tasks[j]){
                res++;
                i++;
            }
            else if(pills>0){
                for(int p=workers.length-1; p>=i; p--){
                    if(workers[p]>=0&&workers[p]+strength>=tasks[j]){
                        workers[p] = -1;
                        tasks[j] = -1;
                        pills--;
                        res++;
                        break;
                    }
                }
            }
            j++;
        }
        return res;

    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 387. 字符串中的第一个唯一字符
    * @DateTime: 7/30/2024 1:33 AM
    * @Params: 
    * @Return 
    */
    public int firstUniqChar(String s) {
        char[] ss = s.toCharArray();
        int[] cnt = new int[26];
        for(int i=0; i<ss.length; i++){
            cnt[ss[i]-'a']++;
        }
        for(int i=0; i<ss.length; i++){
            if(cnt[ss[i]-'a']==1){
                return i;
            }
        }
        return -1;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 321. 拼接最大数
    * @DateTime: 7/29/2024 10:02 PM
    * @Params: 
    * @Return 
    */
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        Deque<Integer> stack1 = new ArrayDeque<>();
        Deque<Integer> stack2 = new ArrayDeque<>();
        int[] max = new int[k];
        for(int i=Math.max(k-nums2.length,0); i<=k&&i<=nums1.length; i++){
            //get largest sub-sequel of nums1
            getLargestSubSequelK(stack1,nums1,i);
            //get largest sub-sequel of nums2
            getLargestSubSequelK(stack2,nums2,k-i);
            //combine two sequels
            int[] combined = combineSequels(stack1, stack2);
            //compare to the max
            max = compare(max,combined)?max:combined;
            System.out.println();
        }
        return max;
    }

    public int[] combineSequels(Deque<Integer> stack1, Deque<Integer> stack2){
        int[] nums1 = new int[stack1.size()];
        for(int i=0; i<nums1.length; i++) nums1[i] = stack1.pollFirst();
        int[] nums2 = new int[stack2.size()];
        for(int i=0; i<nums2.length; i++) nums2[i] = stack2.pollFirst();
        int[] res = new int[nums1.length+nums2.length];
        int i=0;
        int m=0, n=0;
        while(m<nums1.length&&n<nums2.length){
            int p=m,q=n;
            while(p<nums1.length&&q<nums2.length&&nums1[p]==nums2[q]){
                p++;q++;
            }
            if(p>=nums1.length){
                res[i] = nums2[n++];
            }
            else if(q>=nums2.length){
                res[i] = nums1[m++];
            }
            else if(nums1[p]>nums2[q]){
                res[i] = nums1[m++];
            }
            else if(nums1[p]<nums2[q]){
                res[i] = nums2[n++];
            }
            i++;
        }
        while(m<nums1.length){
            res[i++] = nums1[m++];
        }
        while(n<nums2.length){
            res[i++] = nums2[n++];
        }
        return res;
    }
    public boolean compare(int[] nums1, int[] nums2){
        if(nums1.length!=nums2.length){
            return nums1.length>nums2.length;
        }
        for(int i=0; i<nums1.length; i++){
            if(nums1[i]!=nums2[i]){
                return nums1[i]>nums2[i];
            }
        }
        return false;
    }
    public void getLargestSubSequelK(Deque<Integer> stack, int[] nums, int k){
        stack.clear();
        int sub = nums.length-k;
        for(int i=0; i<nums.length; i++){
            while(!stack.isEmpty()&&sub>0&&nums[i]>stack.peekLast()){
                stack.pollLast();
                sub--;
            }
            stack.addLast(nums[i]);
        }
        while (stack.size()>k){
            stack.pollLast();
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 316. 去除重复字母
    * @DateTime: 7/29/2024 4:30 PM
    * @Params: 
    * @Return 
    */
    public String removeDuplicateLetters(String s) {
        boolean[] exist = new boolean[26];
        int[] cnt = new int[26];
        Deque<Character> stack = new ArrayDeque<>();
        char[] ss = s.toCharArray();
        for(int i=0; i<ss.length; i++){
            cnt[ss[i]-'a']++;
        }
        for(int i=0; i<ss.length; i++){
            char cur = ss[i];
            if(exist[cur-'a']){
                cnt[cur-'a']--;
                continue;
            }
            while(!stack.isEmpty()&&cur<stack.peekLast()&&cnt[stack.peekLast()-'a']>0){
                char pre = stack.pollLast();
                exist[pre-'a'] = false;
            }
            stack.add(cur);
            cnt[cur-'a']--;
            exist[cur-'a']=true;
        }
        StringBuilder sb = new StringBuilder();
        while(!stack.isEmpty()){
            sb.append(stack.pollFirst());
        }
        return sb.toString();
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 402. 移掉 K 位数字
    * @DateTime: 7/29/2024 3:56 PM
    * @Params: 
    * @Return 
    */
    public String removeKdigits(String num, int k) {
        char[] stack = new char[num.length()];
        char[] nums = num.toCharArray();
        StringBuilder sb = new StringBuilder();
        int start=0;
        for(int i=0; i<nums.length; i++){
            while(start>0&&stack[start-1]>nums[i]&&k>0){
                start--;
                k--;
            }
            stack[start++] = nums[i];
        }
        start -= k;
        int i=0;
        for(;i<start&&stack[i]=='0';i++);
        if(i==start){
            return "0";
        }
        for(; i<start; i++){
            sb.append(stack[i]);
        }
        return sb.toString();
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 94. 二叉树的中序遍历
    * @DateTime: 7/29/2024 3:30 PM
    * @Params: 
    * @Return 
    */
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> result = new ArrayList<>();
        while(root!=null){
            stack.add(root);
            root = root.left;
        }
        while(!stack.empty()){
            TreeNode cur = stack.pop();
            if(cur!=null){
                result.add(cur.val);
            }
            if(cur.right!=null){
                cur = cur.right;
                while(cur!=null){
                    stack.add(cur);
                    cur = cur.left;
                }
            }
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 20. 有效的括号
    * @DateTime: 7/29/2024 3:19 PM
    * @Params: 
    * @Return 
    */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        Map<Character,Character> map = new HashMap<>();
        map.put('(',')');
        map.put('{','}');
        map.put('[',']');
        for(int i=0; i<s.length(); i++){
            char temp = s.charAt(i);
            if(map.containsKey(temp)){
                stack.add(temp);
            }
            else{
                if(stack.empty()){
                    return false;
                }
                char pre = stack.pop();
                if(map.get(pre)!=temp){
                    return false;
                }
            }
        }
        return stack.empty();
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 73. 矩阵置零
    * @DateTime: 7/29/2024 11:20 AM
    * @Params:
    * @Return
    */
    public void setZeroes(int[][] matrix) {
        boolean row=false,col=false;
        for(int i=0; i<matrix.length; i++){
            if(matrix[i][0]==0){
                col=true;
                break;
            }
        }
        for(int i=0; i<matrix[0].length; i++){
            if(matrix[0][i]==0){
                row=true;
                break;
            }
        }
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                if(matrix[i][j]==0){
                    matrix[i][0]=0;
                    matrix[0][j]=0;
                }
            }
        }
        for(int i=1; i<matrix.length; i++){
            for(int j=1; j<matrix[0].length; j++){
                if(matrix[i][0]==0||matrix[0][j]==0){
                    matrix[i][j] = 0;
                }
            }
        }
        for(int i=0; col&&i<matrix.length; i++){
            matrix[i][0] = 0;
        }
        for(int i=0; row&&i<matrix[0].length; i++){
            matrix[0][i]=0;
        }
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 48. 旋转图像
    * @DateTime: 7/29/2024 11:00 AM
    * @Params:
    * @Return
    */
    public void rotate(int[][] matrix) {
        if(matrix==null||matrix.length==0||matrix.length!=matrix[0].length){
            return;
        }
        int cnt = matrix.length/2;
        for(int i=0;i<cnt;i++){
            int m=i,n=i;
            for(;n<matrix.length-i-1;n++){
                int temp = matrix[m][n];
                matrix[m][n] = matrix[matrix.length-n-1][m];
                matrix[matrix.length-n-1][m] = matrix[matrix.length-m-1][matrix.length-n-1];
                matrix[matrix.length-m-1][matrix.length-n-1] = matrix[n][matrix.length-m-1];
                matrix[n][matrix.length-m-1] = temp;
            }
        }
        System.out.println();
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 289. 生命游戏
    * @DateTime: 7/29/2024 10:13 AM
    * @Params: 
    * @Return 
    */
    public void gameOfLife(int[][] board) {
        if(board==null){
            return;
        }
        int[][] count = new int[board.length][board[0].length];
        for(int i=0;i<board.length;i++){
            for(int j=0; j<board[0].length; j++){
                count[i][j] = sum(board,i,j);
            }
        }
        for(int i=0;i<board.length;i++){
            for(int j=0; j<board[0].length; j++){
                if(board[i][j]==1){
                    if(count[i][j]<2){
                        board[i][j] = 0;
                    }
                    else if(count[i][j]<4){
                        board[i][j] = 1;
                    }
                    else{
                        board[i][j] = 0;
                    }
                }
                else {
                    if(count[i][j]==3){
                        board[i][j] = 1;
                    }
                }
            }
        }
    }
    public int sum(int[][] board, int m, int n){
        int up=Math.max(m-1,0),
                btm=Math.min(m+1,board.length-1),
                l=Math.max(n-1,0),
                r=Math.min(n+1,board[0].length-1);
        int result=0;
        for(int i=up;i<=btm;i++){
            for(int j=l;j<=r;j++){
                result += board[i][j];
            }
        }
        result -= board[m][n];
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 54. 螺旋矩阵
    * @DateTime: 7/29/2024 9:30 AM
    * @Params: 
    * @Return 
    */
    public List<Integer> spiralOrder(int[][] matrix) {
        if(matrix.length==0){
            return new ArrayList<>();
        }
        int circle = Math.min((matrix.length)/2,(matrix[0].length/2));
        int mod=0;
        if(Math.min(matrix.length,matrix[0].length)%2==1){
            if(matrix.length>matrix[0].length){
                mod=1;
            }
            else {
                mod=2;
            }
        }
        int[] pos = new int[]{0,0};
        List<Integer> result = new ArrayList<>();
        for(int i=0; i<circle; i++){
            for(;pos[1]<matrix[0].length-i; pos[1]++){
                result.add(matrix[pos[0]][pos[1]]);
            }
            pos[1]--;
            pos[0]++;
            for(;pos[0]<matrix.length-i; pos[0]++){
                result.add(matrix[pos[0]][pos[1]]);
            }
            pos[0]--;
            pos[1]--;
            for(;pos[1]>=0+i; pos[1]--){
                result.add(matrix[pos[0]][pos[1]]);
            }
            pos[1]++;
            pos[0]--;
            for(;pos[0]>=0+i+1; pos[0]--){
                result.add(matrix[pos[0]][pos[1]]);
            }
            pos[0]++;
            pos[1]++;
        }
        if(mod==2){
            for(;pos[1]<matrix[0].length-circle; pos[1]++){
                result.add(matrix[pos[0]][pos[1]]);
            }
        }
        else if(mod==1){
            for(;pos[0]<matrix.length-circle; pos[0]++){
                result.add(matrix[pos[0]][pos[1]]);
            }
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 506. 相对名次
    * @DateTime: 7/29/2024 8:54 AM
    * @Params: 
    * @Return 
    */
    public String[] findRelativeRanks(int[] score) {
        if(score==null){
            return new String[]{};
        }
        Queue<int[]> queue = new PriorityQueue<>((a,b) -> b[1]-a[1]);
        String[] result = new String[score.length];
        String[] medals = new String[]{"Gold Medal", "Silver Medal", "Bronze Medal"};
        for(int i=0; i<score.length;i++){
            queue.add(new int[]{i,score[i]});
        }
        for(int i=0; i<score.length&&i<3; i++){
            result[queue.poll()[0]] = medals[i];
        }
        for(int i=3; i<score.length; i++){
            result[queue.poll()[0]] = String.valueOf(i+1);
        }
        return result;
    }

    public String[] findRelativeRanks2(int[] score) {
        if(score==null){
            return new String[]{};
        }
        String[] result = new String[score.length];
        String[] medals = new String[]{"Gold Medal", "Silver Medal", "Bronze Medal"};
        int max = 0;
        for(int i=0; i<score.length; i++){
            if(max<score[i]){
                max = score[i];
            }
        }
        int[] arr = new int[max+1];
        for(int i=0; i< score.length; i++){
            arr[score[i]] = i+1;
        }
        int count = 0;
        int i=max;
        for(; i>=0&&count<score.length&&count<3; i--){
            if(arr[i]>0){
                result[arr[i]-1] = medals[count];
                count++;
            }
        }
        for(; i>=0&&count<score.length; i--){
            if(arr[i]>0){
                result[arr[i]-1] = String.valueOf(count+1);
                count++;
            }
        }
        return result;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 120. 三角形最小路径和
    * @DateTime: 7/28/2024 12:57 PM
    * @Params: 
    * @Return 
    */

    public int minimumTotal(List<List<Integer>> triangle) {
        int size = triangle.size();
        boolean[][] visited = new boolean[size][size];
        int[][] minimum = new int[size][size];
        return minimumTotal_BackTrack(triangle,0,0,size-1, minimum, visited);
    }

    public int minimumTotal_BackTrack(List<List<Integer>> triangle, int row, int col, int maxRow, int[][] minimum, boolean[][] visited) {
        if(row==maxRow){
            return triangle.get(row).get(col);
        }
        if(visited[row][col]){
            return minimum[row][col];
        }
        visited[row][col] = true;
        minimum[row][col] = triangle.get(row).get(col)
                +Math.min(minimumTotal_BackTrack(triangle,row+1,col,maxRow, minimum, visited),
                minimumTotal_BackTrack(triangle,row+1,col+1,maxRow,minimum,visited));
        return minimum[row][col];
    }


    /**
    * @Author: Wang Xinxiang
    * @Description: 97. 交错字符串
    * @DateTime: 7/28/2024 11:52 AM
    * @Params: 
    * @Return 
    */
    public boolean isInterleave(String s1, String s2, String s3) {
        int i=s1.length(), j=s2.length();
        if(i+j!=s3.length()){
            return false;
        }
        boolean[][] result = new boolean[i+1][j+1];
        boolean[][] visited = new boolean[i+1][j+1];
        return isInterleave(s1.toCharArray(),s1.length(),
                s2.toCharArray(), s2.length(),
                s3.toCharArray(), result, visited);
    }

    public boolean isInterleave(char[] s1, int i1, char[] s2, int i2, char[] s3, boolean[][] result, boolean[][] visited){
        if(visited[i1][i2]){
            return false;
        }
        visited[i1][i2] = true;
        if(i1+i2==0){
            return true;
        }
        boolean cur = false;
        if(i1>0&&s1[i1-1]==s3[i1+i2-1]){
            cur = cur||isInterleave(s1,i1-1,s2,i2,s3,result,visited);
        }
        if(i2>0&&s2[i2-1]==s3[i1+i2-1]) {
            cur = cur || isInterleave(s1, i1, s2, i2 - 1, s3, result, visited);
        }
        return cur;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 95. 不同的二叉搜索树 II
    * @DateTime: 7/28/2024 11:05 AM
    * @Params: 
    * @Return 
    */
    Object[][] note;
    public List<TreeNode> generateTrees(int n) {
        if(n<1){
            return new ArrayList<TreeNode>();
        }
        note = new Object[n+1][n+1];
        return generateTrees(1,n);
    }

    public List<TreeNode> generateTrees(int l, int r) {
        List<TreeNode> alltrees = new ArrayList<>();
        if(l>r){
            alltrees.add(null);
            return alltrees;
        }
        if(note[l][r]!=null){
            return (List<TreeNode>) note[l][r];
        }
        for(int i=l; i<=r; i++){
            List<TreeNode> leftTrees = generateTrees(l, i-1);
            List<TreeNode> rightTrees = generateTrees(i+1, r);
            for(TreeNode left : leftTrees){
                for(TreeNode right : rightTrees){
                    TreeNode cur = new TreeNode(i);
                    cur.left = left;
                    cur.right = right;
                    alltrees.add(cur);
                }
            }
        }
        note[l][r] = alltrees;
        return alltrees;
    }


    
    /**
    * @Author: Wang Xinxiang
    * @Description:kClosest
    * @DateTime: 12/18/2023 1:28 PM
    * @Params:
    * @Return
    */
    
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq1 = new PriorityQueue<>(new Comparator<int[]>(){
            @Override
            public int compare(int[] a1, int[] a2) {
                int e1 = a1[0]*a1[0]+a1[1]*a1[1];
                int e2 = a2[0]*a2[0]+a2[1]*a2[1];
                return e1-e2;
            }
        });
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b)->((a[0]*a[0]+a[1]*a[1])-(b[0]*b[0]+b[1]*b[1])));
        for(int i=0;i<points.length;i++){
            pq.add(points[i]);
        }
        int[][] result = new int[k][2];
        for(int i=0; i<k; i++){
            result[i] = pq.poll();
        }
        return result;
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: longestPalindrome
    * @DateTime: 12/18/2023 1:12 PM
    * @Params: 
    * @Return 
    */
    public String longestPalindrome(String s) {
        int len = s.length();
        char[] ss = s.toCharArray();
        boolean[][] palindrome = new boolean[len][len];
        int[] res = new int[]{0,0};
        for(int i=0; i<len-1; i++){
            palindrome[i][i] = true;
            palindrome[i][i+1] = ss[i]==ss[i+1];
            if(palindrome[i][i+1]){
                res[0] = i;
                res[1] = i+1;
            }
        }
        palindrome[len-1][len-1] = true;
        for(int i=3; i<=len; i++){
            for(int j=0; j<len-i+1; j++){
                palindrome[j][j+i-1] = palindrome[j+1][j+i-2]&&(ss[j]==ss[j+i-1]);
                if(palindrome[j][j+i-1]){
                    res[0] = j;
                    res[1] = j+i-1;
                }
            }
        }
        return s.substring(res[0], res[1]+1);
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: candyType
    * @DateTime: 12/18/2023 1:04 PM
    * @Params:
    * @Return
    */
    public int distributeCandies(int[] candyType) {
        if(candyType==null||candyType.length==0){
            return 0;
        }
        Arrays.sort(candyType);
        int pre = candyType[0];
        int result = 1;
        for(int i=1; i<candyType.length; i++){
            if(candyType[i]!=pre){
                result++;
            }
            pre = candyType[i];
        }
        int max = candyType.length;
        return Math.min(result, max/2);
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 1493
    * @DateTime: 12/17/2023 5:27 PM
    * @Params:
    * @Return
    */
    public int longestSubarray(int[] nums) {
        int result = 0;
        int zc = 0;
        int l=0, r=0;
        for(; r<nums.length; r++){
            if(nums[r]==0&&++zc>1){
                result = Math.max(result, r-l-zc-1);
                while(nums[l]==1){
                    l++;
                }
                l++;
                zc--;
            }
        }
        result = Math.max(result, r-l-1);
        return result;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 1004
    * @DateTime: 12/16/2023 9:19 PM
    * @Params: 
    * @Return 
    */
    public int longestOnes(int[] nums, int k) {
        int result = 0;
        int zc = 0;
        int l=0, r=0;
        for(; r<nums.length; r++){
            if(nums[r]==0&&++zc>k){
                result = Math.max(result, r-l);
                while(nums[l]==1){
                    l++;
                }
                l++;
                zc--;
            }
        }
        result = Math.max(result, r-l);
        return result;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 1456
    * @DateTime: 12/16/2023 9:11 PM
    * @Params:
    * @Return
    */
    public int maxVowels(String s, int k) {
        if(s.length()<k){
            return -1;
        }
        boolean[] vo = new boolean[256];
        vo['a']=true;
        vo['e']=true;
        vo['i']=true;
        vo['o']=true;
        vo['u']=true;
        char[] ss = s.toCharArray();
        int l=0,r=0;
        int result=0, sum=0;
        for(; r<k; r++){
            if(vo[ss[r]]){
                sum++;
            }
        }
        result=sum;
        for(; r<ss.length; r++,l++){
            if(vo[ss[r]]){
                sum++;
            }
            if(vo[ss[l]]){
                sum--;
            }
            result = Math.max(result, sum);
            if(result==k){
                break;
            }
        }
        return result;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 643
    * @DateTime: 12/15/2023 10:29 PM
    * @Params:
    * @Return
    */
    public double findMaxAverage(int[] nums, int k) {
        if(nums.length<k){
            return -1;
            //throw an error
        }
        int l=0, r=0;
        int sum = 0;
        double result;
        for(; r<k; r++){
            sum += nums[r];
        }
        result = sum;
        for(; r<nums.length; r++, l++){
            sum = sum-nums[l]+nums[r];
            result = Math.max(result, sum);
        }
        return result/k;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 1679
    * @DateTime: 12/15/2023 9:54 PM
    * @Params:
    * @Return
    */
    public int maxOperations(int[] nums, int k) {
        int cnt=0;
        Arrays.sort(nums);
        int l=0, r=nums.length-1;
        while(l<r){
            int sum = nums[l]+nums[r];
            if(sum==k){
                cnt++;
                l++;
                r--;
            }
            else if(sum<k){
                l++;
            }
            else{
                r--;
            }
        }
        return cnt;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 11
    * @DateTime: 12/14/2023 12:47 PM
    * @Params:
    * @Return
    */
    public int maxArea(int[] height) {
        int max_v = 0;
        int l=0, r=height.length-1;
        int cur = 0;
        while(l<r){
            if(height[l]<height[r]){
                cur = height[l]*(r-l);
                l++;
            }
            else{
                cur = height[r]*(r-l);
                r--;
            }
            max_v = Math.max(max_v, cur);
        }
        return max_v;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 392
    * @DateTime: 12/14/2023 12:39 PM
    * @Params:
    * @Return
    */
    public boolean isSubsequence(String s, String t) {
        char[] ss = s.toCharArray();
        char[] ts = t.toCharArray();
        int it = 0;
        for(int i=0; i<ss.length; i++){
            while(it<ts.length&&ts[it]!=ss[i]){
                it++;
            }
            if(it>=ts.length){
                return false;
            }
            it++;
        }
        return true;
    }


    /**
    * @Author: Wang Xinxiang
    * @Description: contest375-2
    * @DateTime: 12/13/2023 6:53 PM
    * @Params:
    * @Return
     *[5,7,8,10,17,18]
    */
    public List<Integer> getGoodIndices(int[][] variables, int target) {
        List result = new ArrayList<Integer>();
        for(int i=-0; i<variables.length; i++){
            long cur = (long)Math.pow(variables[i][0], variables[i][1]);
            cur = cur%10;
            cur = (long)Math.pow(cur, variables[i][2]);
            cur = cur%variables[i][3];
            if(cur==target){
                result.add(i);
            }
        }
        return result;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: contest375-1
    * @DateTime: 12/13/2023 6:49 PM
    * @Params:
    * @Return
    */
    public int countTestedDevices(int[] batteryPercentages) {
        int count = 0;
        for (int i=0; i<batteryPercentages.length; i++){
            if(batteryPercentages[i]>count){
                count++;
            }
        }
        return count;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 443
    * @DateTime: 12/13/2023 4:53 PM
    * @Params:
    * @Return
    */
    public int compress(char[] chars) {
        if(chars==null){
            return 0;
        }
        int len = chars.length;
        int count = 0;
        if(len==1){
            return 1;
        }
        int l=0, r=1, cur=0;
        while(r<len){
            if(chars[r]!=chars[l]){
                chars[cur] = chars[l];
                cur++;
                cur = putNumbers(chars, cur, r-l);
                l=r;
                count++;
            }
            r++;
        }
        chars[cur] = chars[l];
        cur++;
        cur = putNumbers(chars, cur, r-l);
        count++;
        return cur;
    }

    public int putNumbers(char[] chars, int cur, int n){
        if(n==1){
            return cur;
        }
        List<Character> numbers = new ArrayList<>();
        while(n>0){
            numbers.add(0, (char) ('0'+n%10));
            n = n/10;
        }
        for(int i=0; i<numbers.size(); i++){
            chars[cur++] = (char)numbers.get(i);
        }
        return cur;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 334
    * @DateTime: 12/13/2023 4:31 PM
    * @Params: 
    * @Return 
    */
    public boolean increasingTriplet(int[] nums) {
        int left = Integer.MAX_VALUE;
        int right = Integer.MAX_VALUE;
        for(int n:nums){
            if(n<=left){
                left=n;
            }
            else if(n<=right){
                right = n;
            }
            else{
                return true;
            }
        }
        return false;
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 238
    * @DateTime: 12/13/2023 4:15 PM
    * @Params: 
    * @Return 
    */
    public int[] productExceptSelf(int[] nums) {
        if(nums==null){
            return new int[0];
        }
        int len = nums.length;
        int[] products = new int[len];
        products[len-1] = 1;
        for(int i=len-2; i>=0; i--){
            products[i] = products[i+1]*nums[i+1];
        }
        int r = 1;
        for(int i=1; i<len; i++){
            r = r*nums[i-1];
            products[i] = products[i]*r;
        }
        return products;
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 151
    * @DateTime: 12/13/2023 4:18 AM
    * @Params: 
    * @Return 
    */
    public String reverseWords(String s) {
        StringBuilder sb=new StringBuilder();
        int l=s.length()-1,r=s.length()-1;
        s = s.trim();
        while(r>=0){
            l=r;
            while(l>=0&&s.charAt(l)!=' '){
                l--;
            }
            if(l>=-1&&l!=r){
                sb.append(s,l+1,r+1);
                sb.append(" ");
            }
            r=l-1;
        }
        return sb.toString().trim();
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 345
    * @DateTime: 12/13/2023 3:56 AM
    * @Params: 
    * @Return 
    */
    public String reverseVowels(String s) {
        char[] ss = s.toCharArray();
        boolean[] cSet = new boolean[256];
        cSet['a'] = true;
        cSet['e'] = true;
        cSet['i'] = true;
        cSet['o'] = true;
        cSet['u'] = true;
        cSet['A'] = true;
        cSet['E'] = true;
        cSet['I'] = true;
        cSet['O'] = true;
        cSet['U'] = true;
        int l=0, r=s.length()-1;
        while(l<r){
            if(cSet[ss[l]]&&cSet[ss[r]]){
                char a = ss[l];
                ss[l] = ss[r];
                ss[r] = a;
                l++;
                r--;
            }
            else{
                if(!cSet[ss[l]]){
                    l++;
                }
                if(!cSet[ss[r]]){
                    r--;
                }
            }
        }
        return String.copyValueOf(ss);
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 605
    * @DateTime: 12/11/2023 3:18 PM
    * @Params:
    * @Return
    */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int result = 0;
        int l=-1, r=0;
        while(l<flowerbed.length&&r<flowerbed.length){
            while(r<flowerbed.length&&flowerbed[r]==0){
                r++;
            }
            if(r>=flowerbed.length){
                r++;
            }
            result +=getMaxNumbberOfFlowersInInterval(r-l-1);
            l = r+1;
            r = l+1;
        }
        return result>=n;
    }

    public int getMaxNumbberOfFlowersInInterval(int n){
        return n/2;
    }
    /**
    * @Author: Wang Xinxiang
    * @Description: 1431
    * @DateTime: 12/11/2023 3:12 PM
    * @Params:
    * @Return
    */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List result = new ArrayList<Boolean>();
        int max = 0;
        for(int i=0; i<candies.length; i++){
            max = Math.max(max, candies[i]);
        }
        for(int i=0; i<candies.length; i++){
            result.add(candies[i]+extraCandies>=max);
        }
        return result;
    }

    /**
    * @Author: Wang Xinxiang
    * @Description: 1071
    * @DateTime: 12/11/2023 11:45 AM
    * @Params: 
    * @Return 
    */

    public String gcdOfStrings1(String str1, String str2) {
        if(str2.length()>str1.length()) {
            return gcdOfStrings (str2, str1);
        }
        if (str2.equals(str1)){
            return str1;
        }

        if (str1.startsWith(str2)){
            return gcdOfStrings1(str1.substring(str2.length()), str2);
        }

        return "";
    }
    public String gcdOfStrings(String str1, String str2) {
        if(str1==null||str2==null){
            return str1==null?str1:str2;
        }
        char[] strs1 = str1.toCharArray();
        char[] strs2 = str2.toCharArray();
        int len1 = str1.length();
        int len2 = str2.length();
        String subStr1 = findTheLargestConcatenatedSubString(strs1, len1);
        String subStr2 = findTheLargestConcatenatedSubString(strs2, len2);
        len1 = subStr1.length();
        len2 = subStr2.length();
        while(!subStr1.equals(subStr2)){
            if(len1<len2){
                subStr2 = findTheLargestConcatenatedSubString(strs2, len2-1);
                len2 = subStr2.length();
            }
            else{
                subStr1 = findTheLargestConcatenatedSubString(strs1, len1-1);
                len1 = subStr1.length();
            }
        }
        return subStr1;
    }

    public String findTheLargestConcatenatedSubString(char[] str, int cur){
        if(cur==0){
            return "";
        }
        int len = str.length;
        if(len%cur != 0){
            while(len%cur != 0){
                cur--;
            }
            return findTheLargestConcatenatedSubString(str, cur);
        }
        for(int i=0; i<cur; i++){
            for(int j=i+cur; j<len; j=j+cur){
                if(str[j]!=str[i]){
                    return findTheLargestConcatenatedSubString(str, cur-1);
                }
            }
        }
        return String.copyValueOf(str,0,cur);
    }
    
    /**
    * @Author: Wang Xinxiang
    * @Description: 1768
    * @DateTime: 12/11/2023 11:26 AM
    * @Params:
    * @Return
    */
    public String mergeAlternately(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        char[] result = new char[len1+len2];
        char[] words1 = word1.toCharArray();
        char[] words2 = word2.toCharArray();
        int i=0, j=0;
        for(; i<len1&&j<len2; i++, j++){
            result[i+j] = words1[i];
            result[i+j+1] = words2[j];
        }
        while(i<len1){
            result[i+j] = words1[i++];
        }
        while(j<len2){
            result[i+j] = words1[j++];
        }
        return String.valueOf(result);
    }
}
