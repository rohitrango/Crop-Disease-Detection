package com.neoneye.android;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

public class MyPlantsActivity extends AppCompatActivity {

    private ProgressBar progressBar;
    private RecyclerView recyclerView;
    private MyPlantsAdapter myPlantsAdapter;
    private TextView nothing;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_my_plants);

        setTitle("History");

        progressBar = (ProgressBar) findViewById(R.id.myplants_progressbar);
        recyclerView = (RecyclerView) findViewById(R.id.myplants_recyclerview);
        nothing = (TextView) findViewById(R.id.nothing_to_show);

        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getApplicationContext(), LinearLayoutManager.VERTICAL, false);
        recyclerView.setLayoutManager(linearLayoutManager);
        myPlantsAdapter = new MyPlantsAdapter(getApplicationContext());
        recyclerView.setAdapter(myPlantsAdapter);

        if(myPlantsAdapter.getItemCount() == 0) {
            recyclerView.setVisibility(View.GONE);
            nothing.setVisibility(View.VISIBLE);
        }
    }
}
