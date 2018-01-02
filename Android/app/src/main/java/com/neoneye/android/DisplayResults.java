package com.neoneye.android;

import android.content.ContentValues;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.provider.Settings;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class DisplayResults extends AppCompatActivity {

    private RecyclerView recyclerView;
    private Button saveCropButton;
    private Button cancelCropButton;
    private JSONArray data;
    private String finalPath;

    private double lon, lat;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display_results);

        lon = -100;
        lat = -100;

        Intent intent = getIntent();
        String results = intent.getStringExtra("Json");
        finalPath = intent.getStringExtra("image");
//        bitmap = (Bitmap) intent.getParcelableExtra("image");

        Context context = getBaseContext();
        recyclerView = (RecyclerView) findViewById(R.id.crop_list);
        saveCropButton = (Button) findViewById(R.id.save_crop);
        cancelCropButton = (Button) findViewById(R.id.cancel_crop);

        cancelCropButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

        saveCropButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    String path     = finalPath;
                    final String category = data.getJSONObject(0).getString("category_name");
                    final double prob     = data.getJSONObject(0).getDouble("prob");
                    Date currentTime = Calendar.getInstance().getTime();
                    String time = currentTime.toString();

                    // TODO: send an API call to the URL as well
                    RequestQueue queue = Volley.newRequestQueue(getApplicationContext());
                    String url = Constants.save_entry_loc;
                    StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                            new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {

                                }
                            }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            error.printStackTrace();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError {
                            Map<String, String> map = new HashMap<String, String>();
                            map.put("category_name", category);
                            map.put("prob", String.valueOf(prob));
                            map.put("device_ID", getDeviceID(getApplicationContext()));
                            map.put("lat", String.valueOf(lat));
                            map.put("lon", String.valueOf(lon));
                            return map;
                        }
                    };
                    queue.add(stringRequest);

                    Database db = new Database(getApplicationContext());
                    SQLiteDatabase sql = db.getWritableDatabase();
                    ContentValues cv = new ContentValues();
                    cv.put(db.KEY_CATEGORY, category);
                    cv.put(db.KEY_PATH, path);
                    cv.put(db.KEY_PROB, prob);
                    cv.put(db.KEY_TIME, time);
                    sql.insert(db.TABLE, null, cv);
                    Toast.makeText(getApplicationContext(), "Saved successfully.", Toast.LENGTH_LONG).show();
                    finish();

                }
                catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });


        // set up the recycler view
        try {
            // Get the array and put all your stuff there
            data = new JSONArray(results);
            ResultAdapter resultAdapter = new ResultAdapter(data);
            LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getApplicationContext(), LinearLayoutManager.VERTICAL, false);
            recyclerView.setLayoutManager(linearLayoutManager);
            recyclerView.setAdapter(resultAdapter);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* get android_id */
    public static String getDeviceID(Context context) {
        String android_id = Settings.Secure.getString(context.getContentResolver(),
                Settings.Secure.ANDROID_ID);
        return android_id;
    }

    // For saving to internal storage
    private String saveToInternalStorage(Bitmap bitmapImage){
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        Date currentTime = Calendar.getInstance().getTime();
        File mypath = new File(directory, currentTime.toString() + ".png");
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return mypath.getAbsolutePath();
    }

}
