import java.util.*;

/**
 * @Author: Wang Xinxiang
 * @Description:
 * @DateTime: 3/30/2024 8:23 AM
 */

public class DynamicLoadBalancer {

    HashMap<String, Integer> map = new HashMap();
    Random r = new Random();
    List<String> list = new ArrayList<>();
    int ind = 0;
    // 随机返回当前池子中的一个ip
    String get() {
        //todo
        if(ind<=0){
            //Throw exception
            return "";
        }
        int index = r.nextInt(ind);
        String ip = list.get(index);
        while(!map.containsKey(ip)){
            index = r.nextInt(ind);
            ip = list.get(index);
        }
        return ip;
    }

    // 添加一个新的ip
    void add(String ip) {
        //todo
        map.put(ip, ind++);
        list.add(ip);
    }

    // 删除一个指定ip
    void remove(String ip) {
        //todo
        map.remove(ip);
    }
}

