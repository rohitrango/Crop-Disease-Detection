package com.neoneye.android;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Bundle;
import android.support.v4.app.FragmentActivity;
import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.BitmapDescriptorFactory;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.MarkerOptions;

import org.json.JSONArray;
import org.json.JSONObject;

public class MapsActivity extends FragmentActivity implements OnMapReadyCallback {

    private String dummy = "[{'lat' : 61, 'lon' : 32 , 'category_name' : 'HHKAJ', 'healthy' : True}, {'lat' : 64, 'lon' : 32 , 'category_name' : 'jah', 'healthy' : False}, {'lat' : 64, 'lon' : 35 , 'category_name' : 'jah', 'healthy' : False}]";
    private GoogleMap mMap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_maps);
        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;
        try {
            JSONArray plant_list = new JSONArray(dummy);
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
        LatLng center = new LatLng(0,0);
        mMap.moveCamera(CameraUpdateFactory.newLatLng(center));

        LatLng MELBOURNE = new LatLng(-37.813, 144.962);
        mMap.addMarker(new MarkerOptions()
                .position(MELBOURNE)
                .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE)));
    }
}
