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

package client.tools;

import java.net.*;

public class EnumerateHostIPs {
	public static void main(String []args) {
		if(args.length != 1) {
			System.out.println("Provide a single host name on the command line.");
			System.exit(0);
		}
		
		try {
			for(InetAddress add : InetAddress.getAllByName(args[0]))
				System.out.println(add.getHostAddress());
		}
		catch(UnknownHostException e) {
			e.printStackTrace();
		}
	}
}
