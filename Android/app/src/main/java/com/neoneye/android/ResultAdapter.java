package com.neoneye.android;

import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Created by rohitrango on 2/1/18.
 */

public class ResultAdapter extends RecyclerView.Adapter<ResultAdapter.ViewHolder> {

    private JSONArray array;
    public ResultAdapter(JSONArray json) {
        array = json;
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
            double prob = obj.getDouble("prob");
            int probb = (int) (prob*10000);
            prob = probb/100.0;
            holder.prob.setText(String.valueOf(prob) + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public int getItemCount() {
        if(array == null)
            return 0;
        return array.length();
    }

    // Viewholder for the app
    public static class ViewHolder extends RecyclerView.ViewHolder {

        private TextView category;
        private TextView prob;

        public ViewHolder(View itemView) {
            super(itemView);
            category = (TextView) itemView.findViewById(R.id.category);
            prob     = (TextView) itemView.findViewById(R.id.probability);
        }
    }
}
