/**
 * @Author: Wang Xinxiang
 * @Description:
 * @DateTime: 3/30/2024 9:00 AM
 */

public class MyHashSet {
    int size = 9;
    Object[] array = new Object[size];

    int hash(String ip){
        ip = ip.replace('.', '0');//something to adjust
        int n = 0;
        while(size>0){
            n++;
            size = size/10;
        }
        int m = 0;
        int i=n;
        while(n>0){
            m = m*10+(ip.charAt(n)-'0');
        }
        if(array[m]!=null){
            size += 10;
            m = m*10+(ip.charAt(i+1)-'0');
        }
        return m;
    }
}
