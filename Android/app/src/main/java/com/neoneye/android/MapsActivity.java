package com.neoneye.android;

import android.animation.FloatArrayEvaluator;
import android.content.Intent;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Bundle;
import android.support.v4.app.FragmentActivity;
import android.util.Log;
import android.view.View;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.gms.maps.CameraUpdate;
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

    private FloatingActionButton button;
    private String response_to_prob;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_maps);
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);

        Bundle bundle = getIntent().getExtras();
        crop = bundle.getString("plant", "");
        lon = bundle.getDouble("lon");
        lat = bundle.getDouble("lat");
        Log.w("CROP: ", crop);
        Log.w("LON: ", String.valueOf(lon));
        Log.w("LAT: ", String.valueOf(lat));

        button = (FloatingActionButton) findViewById(R.id.fab);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), PlantDangerActivity.class);
                intent.putExtra("response", response_to_prob);
                startActivity(intent);
            }
        });

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
                            response_to_prob = res.getJSONArray("probs").toString();
                            Log.w("response_to_prob", response_to_prob);

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
                                    color = Constants.colors[plant.getInt("index")];
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

        LatLng center = new LatLng(lat+0.001, lon+0.001);
        mMap.addMarker(new MarkerOptions().position(center).title("Current location").icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_GREEN)));
        mMap.moveCamera(CameraUpdateFactory.newLatLng(center));

        CameraUpdate zoom = CameraUpdateFactory.zoomTo(13);
        mMap.animateCamera(zoom);
    }
}
