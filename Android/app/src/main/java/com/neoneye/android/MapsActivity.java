package com.neoneye.android;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Bundle;
import android.support.v4.app.FragmentActivity;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class MapsActivity extends FragmentActivity implements OnMapReadyCallback {

    private String dummy = "[{'lat' : 61, 'lon' : 32 , 'category_name' : 'HHKAJ', 'healthy' : True}, {'lat' : 64, 'lon' : 32 , 'category_name' : 'jah', 'healthy' : False}, {'lat' : 64, 'lon' : 35 , 'category_name' : 'jah', 'healthy' : False}]";
    private GoogleMap mMap;

    private double lon;
    private double lat;
    private String crop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_maps);
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);

        Bundle bundle = getIntent().getExtras();
        crop = bundle.getString("plant", "");
        lon = bundle.getDouble("lon", 0.0);
        lat = bundle.getDouble("lat", 0.0);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;

        RequestQueue queue = Volley.newRequestQueue(getApplicationContext());
        String url = Constants.get_entry;
        StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        // response here
                        try {

                            JSONObject res = new JSONObject(response);

                            JSONArray plant_list = res.getJSONArray("locs");
                            for (int i = 0; i < plant_list.length(); i++) {
                                JSONObject plant = plant_list.getJSONObject(i);
                                Float lat = Float.valueOf(plant.getString("lat"));
                                Float lon = Float.valueOf(plant.getString("lon"));
                                LatLng plantloc = new LatLng(lat, lon);
                                String category = plant.getString("category_name");
                                Float color = null;
                                if(plant.getBoolean("healthy")){
                                    color = BitmapDescriptorFactory.HUE_GREEN;
                                }
                                else{
                                    color = BitmapDescriptorFactory.HUE_RED;
                                }
                                mMap.addMarker(new MarkerOptions().
                                        position(plantloc).
                                        title(category)
                                        .icon(BitmapDescriptorFactory.defaultMarker(color)));
                            }
                        }
                        catch (Exception e){
                            System.out.println(e);
                        }
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
                map.put("crop_name", crop);
                map.put("lat", String.valueOf(lat));
                map.put("lon", String.valueOf(lon));
                return map;
            }
        };
        queue.add(stringRequest);

        LatLng center = new LatLng(lat, lon);
        mMap.moveCamera(CameraUpdateFactory.newLatLng(center));
    }
}
