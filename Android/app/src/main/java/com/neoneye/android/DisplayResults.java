package com.neoneye.android;

import android.content.Context;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.TextView;

import org.json.JSONArray;

public class DisplayResults extends AppCompatActivity {

    private RecyclerView recyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display_results);

        Context context = getBaseContext();
        recyclerView = (RecyclerView) findViewById(R.id.crop_list);

        Intent intent = getIntent();
        String results = intent.getStringExtra("Json");
        try {
            // Get the array and put all your stuff there
            JSONArray data = new JSONArray(results);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
