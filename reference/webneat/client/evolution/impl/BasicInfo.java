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

import client.*;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

abstract class BasicInfo implements Transferable {
	protected BranchIdentifier branch;
	protected long id;
	
	protected BasicInfo(long id) {
		this.branch = BranchIdentifier.getCurrentBranch();
		this.id = id;
	}
	
	public String getBranch() {
		return branch.getName();
	}
	
	public long getId() {
		return id;
	}
	
	public boolean isValid() {
		return id != -1;
	}
	
	public boolean equals(Object other) {
		BasicInfo o = (BasicInfo) other;
		return id == o.id && branch.equals(o.branch);
	}
	
	/**
	 * This method MUST be overriden when equals is overriden! Otherwise, 
	 * hash tables are unusable!
	 * 
	 * @author Jan Prokaj 
	 */
	@Override
	public int hashCode() {
		int hash = 7;
		hash = 31 * hash + (int) (id ^ (id >>> 32));
		hash = 31 * hash + branch.hashCode();
		return hash;
	}
	
	public void load(Element xmlElement) {
		branch = BranchIdentifier.forName(xmlElement.getAttribute("branch"));
		id = Long.parseLong(xmlElement.getAttribute("id"));
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		xmlElement.setAttribute("branch", branch.getName());
		xmlElement.setAttribute("id", Long.toString(id));
	}
	
	public boolean usesCurrentBranch() {
		return branch.isCurrentBranch();
	}
	
	public String toString() {
		return "[" + branch.getName() + "," + Long.toString(id) + "]";
	}
}
