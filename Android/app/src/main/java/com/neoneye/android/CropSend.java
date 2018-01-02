package com.neoneye.android;

import java.io.File;

/**
 * Created by samarjeet on 2/1/18.
 */

public class CropSend {

    public String cropname;
    public File plantimg;

    public CropSend(String cropname, File plantimg) {
        this.cropname = cropname;
        this.plantimg = plantimg;
    }

}
