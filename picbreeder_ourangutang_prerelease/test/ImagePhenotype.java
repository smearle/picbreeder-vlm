/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package test;

import client.ParameterTableInstance;
import client.evolution.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.*;

public class ImagePhenotype extends BufferedImage implements Phenotype {
	private final WritableRaster raster;
	private ImageObserver observer = null;
	private int quality;
	
	public ImagePhenotype() {
		this(ParameterTableInstance.get().getInteger("display", "width"), ParameterTableInstance.get().getInteger("display", "height"));
	}
	
	public ImagePhenotype(int width, int height) {
		super(width, height, TYPE_INT_RGB);
		raster = getRaster();
		quality = Integer.MAX_VALUE;
	}
	
	public double computeInputX(int x) {
		return ((double) ((x << 1) - getWidth() + 1)) / getWidth();
	}
	
	public double computeInputY(int y) {
		return ((double) ((y << 1) - getHeight() + 1)) / getHeight();
	}
	
	public void setObserver(ImageObserver im) {
		observer = im;
	}
	
	public void setValue(int x, int y, double[] value) {
		for(int i = 0; i < value.length; i++)
			value[i] = clamp(value[i] * 255.0);
		raster.setPixel(x, y, value);
	}
	
	public void notifyUpdated() {
		if(observer != null)
			observer.imageUpdate(this, ImageObserver.SOMEBITS, 0, 0, getWidth(), getHeight());
	}
	
	public void notifyCompleted() {
		if(observer != null)
			observer.imageUpdate(this, ImageObserver.ALLBITS, 0, 0, getWidth(), getHeight());
	}
	
	private double clamp(double d){
		return Math.max(0.0, Math.min(255.0, d));
	}
	
	public void save(String file) throws IOException {
		ImageIO.write(this, file.substring(file.lastIndexOf('.') + 1), new File(file));
	}

	public void save(OutputStream os, String format) throws IOException {
		ImageIO.write(this, format, os);
	}
	
	public void setQuality(int quality) {
		this.quality = quality;
	}
	
	public int getQuality() {
		return quality;
	}
}
