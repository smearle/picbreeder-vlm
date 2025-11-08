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

package client.evolution.impl;

import java.util.Map;

final class BranchIdentifier implements Comparable <BranchIdentifier> {
	private String value;
	private static BranchIdentifier CurrentBranch = null;
	private static Map <String, BranchIdentifier> map = null;
	
	public static void beginSession() {
		CurrentBranch = new BranchIdentifier();
		map = new java.util.TreeMap <String, BranchIdentifier> ();
		map.put("", CurrentBranch);
	}
	
	public static void endSession() {
		map.clear();
		map = null;
		CurrentBranch = null;
	}
	
	static {
		beginSession();
	}
	
	public static BranchIdentifier getCurrentBranch() {
		return CurrentBranch;
	}
	
	public static BranchIdentifier forName(String name) {
		if(!map.containsKey(name))
			map.put(name, new BranchIdentifier(name));
		return map.get(name);
	}
	
	private BranchIdentifier() {
		this("");
	}
	
	private BranchIdentifier(String name) {
		value = name;
	}
	
	public void setName(String name) {
		map.remove(value);
		value = name;
		map.put(name, this);
	}
	
	public String getName() {
		return value;
	}
	
	public boolean isCurrentBranch() {
		return value.equals("") || value.equals(CurrentBranch.value);
	}
	
	public int compareTo(BranchIdentifier other) {
		// complicated since current branch can change
		if(isCurrentBranch())
			if(other.isCurrentBranch())
				return 0;
			else
				return -1;
		else if(other.isCurrentBranch())
			return 1;
		else
			return value.compareTo(other.value);
	}
	
	public boolean equals(Object other) {
		if(other instanceof String)
			return value.equals(other);
		else
			return compareTo((BranchIdentifier) other) == 0;
	}
}
