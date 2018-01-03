package com.neoneye.android;

import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Created by rohitrango on 3/1/18.
 */

public class PlantDangerAdapter extends RecyclerView.Adapter<PlantDangerAdapter.ViewHolder> {

    private JSONArray array;
    public PlantDangerAdapter(JSONArray arr) {
        array = arr;
    }
    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.view_item, parent, false);
        ViewHolder h = new ViewHolder(v);
        return h;
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        try {
            JSONObject obj = array.getJSONObject(position);
            holder.category.setText(obj.getString("category_name"));
            double mProb =  obj.getDouble("prob");
            Log.w("mprob", String.valueOf(mProb));
            int t = (int) (mProb*10000);
            Log.w("t", String.valueOf(t));
            mProb = (double)t/100.0;
            Log.w("mprob", String.valueOf(mProb));
            holder.prob.setText(String.valueOf(mProb) + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public int getItemCount() {
        return array.length();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        private TextView category;
        private TextView prob;
        public ViewHolder(View itemView) {
            super(itemView);
            category = itemView.findViewById(R.id.category);
            prob = itemView.findViewById(R.id.probability);
        }
    }
}
