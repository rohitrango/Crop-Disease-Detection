package com.neoneye.android;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.widget.TextView;

import org.json.JSONArray;

public class PlantDangerActivity extends AppCompatActivity {

    public String response;
    public JSONArray probs;

    public TextView nothing_to_show;
    public RecyclerView recyclerView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_plant_danger);
        Intent intent = getIntent();
        response = intent.getStringExtra("response");
        try {
            probs = new JSONArray(response);
        } catch (Exception e) {
            e.printStackTrace();
            probs = new JSONArray();
        }

        this.setTitle("Chances of epidemic");

        nothing_to_show = (TextView) findViewById(R.id.nothing_to_show);
        recyclerView = (RecyclerView) findViewById(R.id.danger_recyclerview);
        recyclerView.setLayoutManager(new LinearLayoutManager(getApplicationContext(), LinearLayoutManager.VERTICAL, false));
        recyclerView.setAdapter(new PlantDangerAdapter(probs));
    }
}
