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

package client;

import java.util.Map;
import java.util.Set;

/**
 * An abstract implementation of the general functionality of the <code>ParameterTable</code>.
 * 
 * @author Nick
 */

public abstract class AbstractParameterTable implements ParameterTable {
	private final Map <String, Map <String, String> > groups;
	private final Map <String, Set <String> > sets;
	private final Map <String, Map <String, Double> > weights;
	
	public AbstractParameterTable() {
		groups = new java.util.TreeMap <String, Map <String, String> > ();
		sets = new java.util.TreeMap <String, java.util.Set <String> > ();
		weights = new java.util.TreeMap <String, Map <String, Double> > ();
	}

	private Map <String, String> getGroup(String groupName) {
		return groups.get(groupName);
	}
	
	private Map <String, String> getOrCreateGroup(String groupName) {
		if(!groups.containsKey(groupName)) {
			Map <String, String> m = new java.util.TreeMap <String, String> ();
			groups.put(groupName, m);
			return m;
		}
		else
			return getGroup(groupName);
	}
	
	private Set <String> getSet(String setName) {
		return sets.get(setName);
	}
	
	public String [] getSetAsArray(String setName) {
		Set <String> set = getSet(setName);
		if(set == null)
			return null;
		
		String [] type = {};
		return set.toArray(type);
	}
	
	private Set <String> getOrCreateSet(String setName) {
		if(!sets.containsKey(setName)) {
			Set <String> s = new java.util.TreeSet <String> ();
			sets.put(setName, s);
			weights.put(setName, new java.util.TreeMap <String, Double> ());
			return s;
		}
		else
			return sets.get(setName);
	}
	
	public double getWeightOfSetItem(String setName, String item) {
		Map <String, Double> m = weights.get(setName);
		if(m == null)
			return 0;
		
		if(!m.containsKey(item))
			return 0;
		
		return m.get(item);
	}

	public void addItemToSet(String setName, String item, double weight) {
		//System.out.println("Parameter [" + setName + "]+=" + item);
		getOrCreateSet(setName).add(item);
		weights.get(setName).put(item, weight);
	}
	
	public void addItemToSet(String setName, String item) {
		addItemToSet(setName, item, 1.0);
	}
	
	public String getRandomItemFromSet(String setName) {
		Set <String> set = getSet(setName);
		if(set == null)
			return null;
		
		Map <String, Double> m = weights.get(setName);
		
		double sum = 0.0;
		for(String s : set)
			sum += m.get(s);
		
		double r = client.utilities.Random.instance().nextDouble() * sum;
		
		sum = 0.0;
		for(String s : set) {
			sum += m.get(s);
			
			if(sum > r)
				return s;
		}
		
		return set.iterator().next();
	}

	
	public void setParameter(String groupName, String parameterName, String value) {
		//System.out.println("Parameter [" + groupName + ":" + parameterName + "]=" + value);
		getOrCreateGroup(groupName).put(parameterName, value);
	}
	
	public String getParameter(String groupName, String parameterName) {
		Map <String, String> m = getGroup(groupName);
		
		if(m == null)
			return null;
		else
			return m.get(parameterName);
	}
	
	public final double getDouble(String groupName, String parameterName) {
		return Double.parseDouble(getParameter(groupName, parameterName));
	}
	
	public final int getInteger(String groupName, String parameterName) {
		return Integer.parseInt(getParameter(groupName, parameterName));
	}
	
	public final boolean getBoolean(String groupName, String parameterName) {
		return Boolean.parseBoolean(getParameter(groupName, parameterName));
	}
}
