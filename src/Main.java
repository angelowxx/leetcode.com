import javax.xml.transform.SourceLocator;
import java.util.*;

public class Main {
    public static char[][] convertArrayChar(){
        Scanner sc = new Scanner(System.in);
        String s = new String();
        while(sc.hasNextLine()){
            s = sc.nextLine();
            String[] ss = s.split("],");
            for(int i=0; i<ss.length; i++){
                ss[i] = ss[i].replace("[", "")
                        .replace(",","")
                        .replace("]","")
                        .replace("\"","");
            }
            char[][] nn = new char[ss.length][ss[0].length()];
            for(int i=0; i<ss.length; i++){
                for(int j=0; j<ss[0].length(); j++){
                    nn[i][j] = ss[i].charAt(j);
                }
            }
            return nn;
        }
        return null;
    }
    public static int[][] convertArray(){
        Scanner sc = new Scanner(System.in);
        String s = new String();
        while(sc.hasNextLine()){
            s = sc.nextLine();
            String[] ss = s.split("],");
            for(int i=0; i<ss.length; i++){
                ss[i] = ss[i].replace("[", "")
                        .replace(",","")
                        .replace("]","");
            }
            int[][] nn = new int[ss.length][ss[0].length()];
            for(int i=0; i<ss.length; i++){
                for(int j=0; j<ss[0].length(); j++){
                    nn[i][j] = ss[i].charAt(j)-'0';
                }
            }
            return nn;
        }
        return null;
    }
    public static void main(String[] args) {
        Solution solution = new Solution();
        char[][] board = new char[][]{
                {'.', '.', '9', '7', '4', '8', '.', '.', '.'},
                {'7', '.', '.', '.', '.', '.', '.', '.', '.'},
                {'.', '2', '.', '1', '.', '9', '.', '.', '.'},
                {'.', '.', '7', '.', '.', '.', '2', '4', '.'},
                {'.', '6', '4', '.', '1', '.', '5', '9', '.'},
                {'.', '9', '8', '.', '.', '.', '3', '.', '.'},
                {'.', '.', '.', '8', '.', '3', '.', '2', '.'},
                {'.', '.', '.', '.', '.', '.', '.', '.', '6'},
                {'.', '.', '.', '2', '7', '5', '9', '.', '.'}
        };
        solution.solveSudoku(board);
    }

    public static List<int[]> getIndexWithSum(List<Integer> list, int num){
        List<int[]> res = new ArrayList<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i=0; i<list.size(); i++){
            int n = list.get(i);
            int target = num-n;
            List<Integer> idxList = new ArrayList<>();
            if(map.containsKey(target)){
                idxList = map.get(target);
                int[] pairIdx = new int[2];
                pairIdx[0] = i;
                pairIdx[1] = idxList.get(0);
                System.out.println(pairIdx[0]+","+pairIdx[1]);
                res.add(pairIdx);
                idxList.remove(0);
                if(idxList.size()==0){
                    map.remove(n);
                }
            }
            else{
                idxList = map.getOrDefault(n, new ArrayList<Integer>());
                idxList.add(i);
                map.put(n, idxList);
            }
        }
        return res;

    }

    public static String convertArray2String(int[] n){
        if(n==null||n.length==0){
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<n.length-1; i++){
            sb.append(n[i]);
            sb.append(',');
        }
        sb.append(n[n.length-1]);
        return sb.toString();
    }

    public static String bigDigitPlus(String s1, String s2){
        char[] ss1 = s1.toCharArray();
        char[] ss2 = s2.toCharArray();
        StringBuilder sb = new StringBuilder();
        int i = s1.length()-1, j = s2.length()-1;
        int jump = 0;
        while(i>=0&&j>=0){
            int cur = ss1[i]-'0'+ss2[j]-'0'+jump;
            jump = cur/10;
            cur = cur%10;
            sb.insert(0,(char)(cur+'0'));
            i--;
            j--;
        }
        while(i>=0){
            int cur = ss1[i]-'0'+jump;
            jump = cur/10;
            cur = cur%10;
            sb.insert(0,cur+'0');
            i--;
        }
        while(j>=0){
            int cur = ss2[j]-'0'+jump;
            jump = cur/10;
            cur = cur%10;
            sb.insert(0,cur+'0');
            j--;
        }
        if(jump!=0){
            sb.insert(0,jump+'0');
        }
        return sb.toString();
    }

    public static void mergeSort(String[] ss, int l, int r){
        if(l>=r){
            return;
        }
        int mid = (l+r)/2;
        mergeSort(ss,l,mid);
        mergeSort(ss,mid+1,r);
        int i=l, j=mid+1, p=0;
        String[] temp = new String[r-l+1];
        for(; i<=mid&&j<=r;p++){
            if(!compare(ss[i],ss[j])){
                temp[p] = ss[i++];
            }
            else{
                temp[p] = ss[j++];
            }

        }
        while(i<=mid){
            temp[p++] = ss[i++];
        }
        while(j<=r){
            temp[p++] = ss[j++];
        }
        for(int q=l; q<=r; q++){
            ss[q] = temp[q-l];
        }
    }

    public static void swap(String[] ss, int i, int j){
        String temp = ss[i];
        ss[i] = ss[j];
        ss[j] = temp;
    }



    public static boolean compare(String s, String t1) {
        int i=0;
        for(; i<s.length()&&i<t1.length(); i++){
            if(s.charAt(i)<t1.charAt(i)){
                return false;
            }
            else{
                return true;
            }
        }
        if(i==s.length()){
            return false;
        }
        else{
            return true;
        }
    }

    private static int countSubstrings(String t, String sBegin, String sEnd) {
        //todo
        HashSet<String> set = new HashSet<>();
        List<Integer> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();
        matchAll(t, sBegin, l1);
        matchAll(t, sEnd, l2);
        List<int[]> result = new ArrayList<>();
        for(int i=0, j=0; i<l1.size()&&j<l2.size();){
            if(l1.get(i)<=l2.get(j)){
                for(int n=j; n<l2.size(); n++){
                    result.add(new int[]{l1.get(i),l2.get(n)});
                }
                i++;
            }
            else{
                j++;
            }
        }
        int count = 0;
        for(int[] a: result){
            String temp = t.substring(a[0], a[1]);
            if(!set.contains(temp)){
                count++;
                set.add(temp);
            }
        }
        return count;
    }

    private static void matchAll(String t, String match, List<Integer> t_n){
        for(int i=0; i<=t.length()-match.length(); i++){
            if(matchOne(t, match, i)){
                t_n.add(i);
            }
        }
    }
    private static boolean matchOne(String t, String match, int index){
        int length = match.length();
        if(length>t.length()-index){
            return false;
        }
        for(int i=0; i<length; i++){
            if(t.charAt(index+i)!=match.charAt(i)){
                return false;
            }
        }
        return true;
    }
}