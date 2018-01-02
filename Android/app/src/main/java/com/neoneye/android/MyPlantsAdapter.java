package com.neoneye.android;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.StringTokenizer;

/**
 * Created by rohitrango on 2/1/18.
 */

public class MyPlantsAdapter extends RecyclerView.Adapter<MyPlantsAdapter.ViewHolder> {

    private Context ctx;
    private ArrayList<MyPlants> myPlants;

    MyPlantsAdapter(Context ctx) {
        this.ctx = ctx;
        Database sql = new Database(ctx);
        SQLiteDatabase db = (sql).getReadableDatabase();
        Cursor c = db.rawQuery("SELECT * FROM " + sql.TABLE + " ORDER BY " + sql.KEY_ID + " DESC;", null);
        myPlants = new ArrayList<MyPlants>();
        if(c.moveToFirst()) {
            do {
                MyPlants p = new MyPlants();
                p.path = c.getString(c.getColumnIndex(sql.KEY_PATH));
                p.category = c.getString(c.getColumnIndex(sql.KEY_CATEGORY));
                p.prob = c.getDouble(c.getColumnIndex(sql.KEY_PROB));
                p.date = new Date(c.getString(c.getColumnIndex(sql.KEY_TIME)));
                p.lon = c.getDouble(c.getColumnIndex(sql.KEY_LON));
                p.lat = c.getDouble(c.getColumnIndex(sql.KEY_LAT));
                Log.e("PATH", p.path);
                myPlants.add(p);
            } while(c.moveToNext());
        }
        c.close();
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.plant_item, parent, false);
        ViewHolder h = new ViewHolder(v);
        return h;
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        final MyPlants p = myPlants.get(position);
        Bitmap b = BitmapFactory.decodeFile(p.path);
        if(b != null) {
            Log.e("ok", b.toString());
            holder.cropView.setImageBitmap(b);
        }

        holder.category.setText(p.category);
        holder.time.setText(p.date.toString());
        int t = (int) (p.prob*10000);
        double d = t/100.0;
        holder.prob.setText(String.valueOf(d) + "%");
        holder.linearLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(ctx, MapsActivity.class);
                String plant = p.category.split("__")[0];
                intent.putExtra("plant", plant);
                intent.putExtra("lon", p.lon);
                intent.putExtra("lat", p.lat);
                ctx.startActivity(intent);
            }
        });
    }

    @Override
    public int getItemCount() {
        return myPlants.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        public ImageView cropView;
        public TextView category;
        public TextView time;
        public TextView prob;
        public LinearLayout linearLayout;

        public ViewHolder(View itemView) {
            super(itemView);
            cropView = itemView.findViewById(R.id.crop_image);
            category = itemView.findViewById(R.id.category);
            time     = itemView.findViewById(R.id.time);
            prob     = itemView.findViewById(R.id.probability);
            linearLayout = itemView.findViewById(R.id.main_clicker);
        }
    }
}
