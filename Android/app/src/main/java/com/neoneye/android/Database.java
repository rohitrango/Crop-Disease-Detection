package com.neoneye.android;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

/**
 * Created by rohitrango on 2/1/18.
 */
public class Database extends SQLiteOpenHelper {

    // All Static variables
    // Database Version
    private static final int DATABASE_VERSION = 4;

    // Database Name
    public static final String DATABASE_NAME = "crops";

    // Contacts table name
    public static final String TABLE = "crops";

    // Contacts Table Columns names
    public static final String KEY_CATEGORY = "category";
    public static final String KEY_PATH = "path";
    public static final String KEY_PROB = "prob";
    public static final String KEY_TIME = "time";
    public static final String KEY_ID   = "id";

    public static final String KEY_LON  = "lon";
    public static final String KEY_LAT  = "lat";


    public Database(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    // Creating Tables
    @Override
    public void onCreate(SQLiteDatabase db) {
        String CREATE_CONTACTS_TABLE = "CREATE TABLE " + TABLE + "("
                + KEY_PATH + " TEXT," + KEY_CATEGORY + " TEXT," + KEY_TIME + " TEXT," + KEY_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                + KEY_LAT  + " DOUBLE, " + KEY_LON + " DOUBLE,"
                + KEY_PROB + " DOUBLE" + ")";
        db.execSQL(CREATE_CONTACTS_TABLE);
    }

    // Upgrading database
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // Drop older table if existed
        db.execSQL("DROP TABLE IF EXISTS " + TABLE);
        // Create tables again
        onCreate(db);
    }
}