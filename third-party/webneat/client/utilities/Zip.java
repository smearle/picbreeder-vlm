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

package client.utilities;

import java.util.zip.*;
import java.io.*;

/**
 * Methods to compress and decompress text.  This should cut down on the
 * required bandwidth and storage space.
 * <p>
 * http://java.sun.com/j2se/1.5.0/docs/api/java/util/zip/Deflater.html
 * 
 * @author Nick
 */

public class Zip {
	private final static int BUFFER_SIZE = 1 << 10; // 1 KB
	
	/**
	 * Compresses the information stored in the data string using the
	 * default ZLIB deflate routine.
	 * 
	 * @param data The information to compress
	 * @return The compressed buffer
	 */
	public static byte []compressText(String data) {
		Deflater deflater = new Deflater();
		deflater.setInput(data.getBytes());
		deflater.finish();
		
		byte [] buffer = new byte[BUFFER_SIZE];
		int read = -1;
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		
		while(!deflater.finished()) {
			read = deflater.deflate(buffer);
			bytes.write(buffer, 0, read);
		}
		
		return bytes.toByteArray();
	}
	
	/**
	 * Decompresses the data buffer and stores the result as a string.
	 * <p>
	 * This method assumes that a string was encoded! Do not try to decompress
	 * arbitrary buffers with this method.
	 * 
	 * @param data The compressed data
	 * @return The uncompressed text
	 * @throws DataFormatException
	 */
	public static String decompressText(byte []data)
			throws DataFormatException {
		Inflater inflater = new Inflater();
		inflater.setInput(data);
		
		byte [] buffer = new byte[BUFFER_SIZE];
		int read = -1;
		
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		while(!inflater.finished()) {
			read = inflater.inflate(buffer);
			bytes.write(buffer, 0, read);
		}
		
		return bytes.toString();
	}
	
	/**
	 * Compresses the information stored in the data string so that it can be manipulated with standard zip utilites
	 * 
	 * @param toBeCompressed The information to compress
	 * @return The compressed buffer
	 */
	public static byte[] zip(String toBeCompressed)
	{
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		byte[] value={};
		try{
			ZipOutputStream zos=new ZipOutputStream(bytes);
			zos.putNextEntry(new ZipEntry("a"));
			zos.write(toBeCompressed.getBytes());
			zos.closeEntry();
			zos.close();
			value=bytes.toByteArray();
		}catch(Exception e){System.err.println("Error zipping up: Check Status of beans and franks");}
		return value;
	}
	/**
	 * Decompresses the data buffer and stores the result as a string.
	 * <p>
	 * This method assumes that a string was encoded! Do not try to decompress
	 * arbitrary buffers with this method.
	 * 
	 * @param toBeUncompressed The compressed data
	 * @return The uncompressed text
	 */
	public static String unzip(byte[] toBeUncompressed)
	{
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		ByteArrayInputStream info=new ByteArrayInputStream(toBeUncompressed);
		byte[] value={};
		try{
			ZipInputStream zis=new ZipInputStream(info);
			zis.getNextEntry();
			byte[] buf = new byte[1024];
	        int len;
	        while ((len = zis.read(buf)) > 0) {
	            bytes.write(buf, 0, len);
	        }
			value=bytes.toByteArray();
		}catch(Exception e){System.err.println("Error unzipping: She probably didn't want it anyways.");}
		return new String(value);
	}
}
