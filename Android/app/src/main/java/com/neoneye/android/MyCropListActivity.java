package com.neoneye.android;

import android.content.Context;
import android.provider.Settings;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.text.Layout;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.HashMap;
import java.util.Map;

public class MyCropListActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private Button cancelBtn;
    private Button saveBtn;

    private boolean[] selected;
    private String[] names;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_my_crop_list);
        names = getResources().getStringArray(R.array.my_crops);
        selected = new boolean[names.length];
        for (int i=0;i<names.length; i++)
            selected[i] = false;

        cancelBtn = (Button) findViewById(R.id.cancel_button);
        saveBtn   = (Button) findViewById(R.id.save_button);

        cancelBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });
        setTitle("My Crops");

        saveBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                final Map<String, String> map = new HashMap<String, String>();
                map.put("deviceID", getDeviceID(getApplicationContext()));
                for(int i=0;i<names.length; i++) {
                    if(selected[i] == true) {
                        map.put(names[i], "1");
                        Log.w("CROP:", names[i]);
                    }
                }

                /* send request */
                RequestQueue queue = Volley.newRequestQueue(getApplicationContext());
                String url = Constants.update_crops;
                StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {
                                Toast.makeText(getApplicationContext(), "Updated successfully.", Toast.LENGTH_SHORT).show();
                            }
                        }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        error.printStackTrace();
                    }
                }){
                    @Override
                    protected Map<String, String> getParams() throws AuthFailureError {
                        return map;
                    }
                };
                queue.add(stringRequest);
                finish();
            }
        });

        recyclerView = (RecyclerView) findViewById(R.id.my_crops_recyclerview);
        LinearLayoutManager layoutManager = new LinearLayoutManager(getApplicationContext(), LinearLayoutManager.VERTICAL, false);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setAdapter(new CropAdapter());
    }


    /* get android_id */
    public static String getDeviceID(Context context) {
        String android_id = Settings.Secure.getString(context.getContentResolver(),
                Settings.Secure.ANDROID_ID);
        return android_id;
    }

    /* adapter here */
    public class CropAdapter extends RecyclerView.Adapter<CropAdapter.ViewHolder> {

        public String[] names;
        public CropAdapter() {
            names = getResources().getStringArray(R.array.my_crops);
            Log.w("names", String.valueOf(names.length));
        }

        @Override
        public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            LayoutInflater inflater = LayoutInflater.from(parent.getContext());
            View v = inflater.inflate(R.layout.crop_layout, parent, false);
            ViewHolder h = new ViewHolder(v);
            return h;
        }

        @Override
        public void onBindViewHolder(ViewHolder holder, final int position) {
            String name = names[position];
            holder.ckbox.setHint(name);
            holder.ckbox.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    selected[position] = !selected[position];
                }
            });
        }

        @Override
        public int getItemCount() {
            return names.length;
        }

        public class ViewHolder extends RecyclerView.ViewHolder{

            public CheckBox ckbox;

            public ViewHolder(View itemView) {
                super(itemView);
                ckbox = (CheckBox) itemView.findViewById(R.id.ckbox);
            }
        }
    }

}
