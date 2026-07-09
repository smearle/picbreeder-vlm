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

import java.security.MessageDigest;

/**
 * Provides a means to create a MD5 hash from a password so that
 * all transmission to the server are safe.
 * 
 * @author Nick
 */
public class PasswordEncoder {
	private static final MessageDigest encoder;
	
	static {
		MessageDigest temp = null;
		
		try {
			temp = MessageDigest.getInstance("MD5");
		}
		catch(java.security.NoSuchAlgorithmException e) {
		}
		finally {
			encoder = temp;
		}
	}
	
	/**
	 * Encodes a password with the MD5 function.
	 * 
	 * @param password The password
	 * @return The MD5 sum for the password
	 */
	public static String encode(String password) {
		byte [] b = password.getBytes();
		encoder.update(b);
		String p = bytesToString(encoder.digest());

		java.util.Arrays.fill(b, 0, b.length, (byte)0);
		b = null;
		
		return p;
	}

	/**
	 * Encodes a password with the MD5 function. This
	 * method is safer then the String variant, since
	 * char arrays do not need to be interned.
	 * <p>
	 * This method works well with the javax.swing.JPasswordField.
	 * You should use java.util.Arrays.fill to remove the
	 * data from memory.
	 * 
	 * @param password The password
	 * @return The MD5 sum for the password
	 */
	public static String encode(char []password) {
		byte[]b = new byte[password.length];
		for(int i = 0; i < b.length; i++)
			b[i] = (byte) password[i];
		
		encoder.update(b);
		String p = bytesToString(encoder.digest());
		
		java.util.Arrays.fill(b, 0, b.length, (byte)0);
		b = null;
		
		return p;
	}
	
	private static final String map = "0123456789abcdef";
	
	private static String bytesToString(byte[]data) {
		String r = "";
		
		for(byte b : data)
			r += map.charAt((b >> 4) & 0xf) + "" + map.charAt(b & 0xf);
		
		return r;
	}
	
	private static void unitTest() throws client.UnitTestFailed {
		String p = "cppn06";
		String r = encode(p);
		
		if(!r.equals("b7a3a59292b5a8206e3448e0dfe2b360"))
			throw new client.UnitTestFailed(PasswordEncoder.class);

		char []p2= {'c', 'p', 'p', 'n', '0', '6'};
		String r2 = encode(p2);
		
		if(!r2.equals("b7a3a59292b5a8206e3448e0dfe2b360"))
			throw new client.UnitTestFailed(PasswordEncoder.class);

	}
}
